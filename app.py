import os
import dotenv
import uuid
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename
from pypdf import PdfReader

dotenv.load_dotenv()

from core import vectorstore, chat_history
from core.rag_pipeline import ingest_document, answer_query
from core.audio_generator import generate_audio_overview
from core.config import (UPLOAD_FOLDER, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA,
                         CHUNK_SIZE, CHUNK_OVERLAP, AUDIO_CACHE_DIR, RAG_MODE,
                         MAX_HOSTED_SOURCES)

app = Flask(__name__)
if RAG_MODE == "hosted":
    app.secret_key = os.getenv("FLASK_SECRET_KEY")
    if not app.secret_key:
        raise RuntimeError("FLASK_SECRET_KEY is required in hosted mode.")
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
    app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Warm the embedding model at startup so the first query isn't slow.
if RAG_MODE == "local":
    try:
        from core.embeddings import get_embedding_model
        get_embedding_model()
    except Exception as _e:
        print(f"[startup] Embedding model warm-up skipped: {_e}")


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
DEFAULT_NOTEBOOK = "default_kb"


@app.before_request
def ensure_workspace():
    if RAG_MODE == "hosted" and "workspace_id" not in session:
        session.permanent = True
        session["workspace_id"] = uuid.uuid4().hex


def scoped_notebook(name: str = None) -> str:
    name = (name or DEFAULT_NOTEBOOK).strip() or DEFAULT_NOTEBOOK
    name = name[:100]
    return f"{session['workspace_id']}:{name}" if RAG_MODE == "hosted" else name


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_top_k=DEFAULT_TOP_K,
        default_alpha=DEFAULT_HYBRID_ALPHA,
        default_chunk_size=CHUNK_SIZE,
        default_chunk_overlap=CHUNK_OVERLAP,
        default_notebook=DEFAULT_NOTEBOOK,
        rag_mode=RAG_MODE,
    )


# ----------------------------------------------------------------
# Sources (documents inside a notebook)
# ----------------------------------------------------------------

@app.route("/api/sources", methods=["GET"])
def get_sources():
    notebook = scoped_notebook(request.args.get("notebook"))
    return jsonify({"status": "success", "sources": vectorstore.list_sources(notebook)})


@app.route("/api/sources/upload", methods=["POST"])
def upload_sources():
    """Accepts one or more files in the 'files' field and indexes each."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"status": "error", "message": "No files selected."}), 400

    notebook = scoped_notebook(request.form.get("notebook"))
    api_key = (request.form.get("gemini_api_key") or "").strip() or None
    if RAG_MODE == "hosted" and not (api_key or os.getenv("GEMINI_API_KEY")):
        return jsonify({"status": "error", "message": "Enter a Gemini API key before uploading."}), 400
    chunk_size = request.form.get("chunk_size", type=int)
    chunk_overlap = request.form.get("chunk_overlap", type=int)

    results = []
    existing_sources = {item["source"] for item in vectorstore.list_sources(notebook)} if RAG_MODE == "hosted" else set()
    for uploaded_file in files:
        if uploaded_file.filename == "":
            continue
        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": uploaded_file.filename, "status": "error",
                             "message": "Unsupported file type."})
            continue

        filename = secure_filename(uploaded_file.filename)
        if not filename:
            results.append({"filename": uploaded_file.filename, "status": "error",
                             "message": "Please use a filename with letters or numbers."})
            continue
        if RAG_MODE == "hosted" and filename not in existing_sources and len(existing_sources) >= MAX_HOSTED_SOURCES:
            results.append({"filename": filename, "status": "error",
                             "message": f"Demo limit: {MAX_HOSTED_SOURCES} sources per notebook."})
            continue
        file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")

        try:
            uploaded_file.save(file_path)
            text = extract_text(file_path)
            if not text.strip():
                results.append({"filename": filename, "status": "error",
                                 "message": "No extractable text found."})
                continue

            chunk_count = ingest_document(
                text=text, source=filename, collection_name=notebook,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, api_key=api_key,
            )
            results.append({"filename": filename, "status": "success", "chunk_count": chunk_count})
            existing_sources.add(filename)

        except Exception as e:
            message = str(e) if RAG_MODE == "local" or isinstance(e, ValueError) else "Could not index this file. Check your Gemini quota and Supabase connection."
            results.append({"filename": uploaded_file.filename, "status": "error", "message": message})
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return jsonify({"status": "success", "results": results,
                     "sources": vectorstore.list_sources(notebook)})


@app.route("/api/sources/<path:source_name>", methods=["DELETE"])
def delete_source(source_name):
    notebook = scoped_notebook(request.args.get("notebook"))
    removed = vectorstore.delete_source(notebook, source_name)
    return jsonify({"status": "success", "message": f"Removed {removed} chunks from '{source_name}'."})


# ----------------------------------------------------------------
# Chat
# ----------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    query_text = (data.get("query") or "").strip()
    if not query_text:
        return jsonify({"status": "error", "message": "Message cannot be empty."}), 400

    notebook = scoped_notebook(data.get("notebook"))
    top_k = int(data.get("top_k", DEFAULT_TOP_K))
    alpha = float(data.get("alpha", DEFAULT_HYBRID_ALPHA))
    api_key = (data.get("gemini_api_key") or "").strip() or None

    try:
        result = answer_query(notebook, query_text, top_k=top_k, alpha=alpha, api_key=api_key)
        return jsonify({
            "status": "success",
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
        })
    except Exception as e:
        message = str(e) if RAG_MODE == "local" else "Could not answer right now. Check your Gemini quota and Supabase connection."
        return jsonify({"status": "error", "message": message}), 500


@app.route("/api/chat/history", methods=["GET"])
def get_chat_history():
    notebook = scoped_notebook(request.args.get("notebook"))
    return jsonify({"status": "success", "history": chat_history.get_history(notebook)})


@app.route("/api/chat/history", methods=["DELETE"])
def reset_chat_history():
    notebook = scoped_notebook(request.args.get("notebook"))
    chat_history.clear_history(notebook)
    return jsonify({"status": "success", "message": "Conversation cleared."})


# ----------------------------------------------------------------
# Audio overview (podcast / narrator)
# ----------------------------------------------------------------

@app.route("/api/audio/generate", methods=["POST"])
def generate_audio():
    data = request.get_json(silent=True) or {}

    notebook = scoped_notebook(data.get("notebook"))
    mode = (data.get("mode") or "narrator").strip()
    topic = (data.get("topic") or "").strip()
    api_key = (data.get("gemini_api_key") or "").strip() or None

    if mode not in ("podcast", "narrator"):
        return jsonify({"status": "error", "message": "mode must be 'podcast' or 'narrator'."}), 400

    if not topic:
        return jsonify({"status": "error", "message": "Please enter a topic or question for the audio to focus on."}), 400

    if not api_key:
        return jsonify({"status": "error", "message": "Please enter your Gemini API key in the left panel before generating audio."}), 400

    sources = vectorstore.list_sources(notebook)
    if not sources:
        return jsonify({"status": "error", "message": "No sources found. Please upload at least one document before generating audio."}), 400

    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(AUDIO_CACHE_DIR, filename)

    try:
        generate_audio_overview(notebook, mode, output_path, topic=topic, api_key=api_key)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception:
        return jsonify({"status": "error", "message": "Something went wrong while generating audio. Please check your API key and try again."}), 500

    return jsonify({"status": "success", "audio_url": f"/api/audio/{filename}"})


@app.route("/api/audio/<path:filename>", methods=["GET"])
def serve_audio(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(AUDIO_CACHE_DIR, safe_name)
    if not os.path.isfile(file_path):
        return jsonify({"status": "error", "message": "Audio file not found."}), 404
    return send_file(file_path, mimetype="audio/mpeg")


# ----------------------------------------------------------------
# Notebooks (collections)
# ----------------------------------------------------------------

@app.route("/api/notebooks", methods=["GET"])
def list_notebooks():
    notebooks = vectorstore.list_collections()
    if RAG_MODE == "hosted":
        prefix = session["workspace_id"] + ":"
        notebooks = [name[len(prefix):] for name in notebooks if name.startswith(prefix)]
    return jsonify({"status": "success", "notebooks": notebooks})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", host="0.0.0.0", port=port)
