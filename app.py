import os
import dotenv
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pypdf import PdfReader

from core import vectorstore, chat_history
from core.rag_pipeline import ingest_document, answer_query
from core.config import UPLOAD_FOLDER, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA, CHUNK_SIZE, CHUNK_OVERLAP

dotenv.load_dotenv()

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
DEFAULT_NOTEBOOK = "default_kb"


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
    )


# ----------------------------------------------------------------
# Sources (documents inside a notebook)
# ----------------------------------------------------------------

@app.route("/api/sources", methods=["GET"])
def get_sources():
    notebook = request.args.get("notebook", DEFAULT_NOTEBOOK)
    return jsonify({"status": "success", "sources": vectorstore.list_sources(notebook)})


@app.route("/api/sources/upload", methods=["POST"])
def upload_sources():
    """Accepts one or more files in the 'files' field and indexes each."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"status": "error", "message": "No files selected."}), 400

    notebook = request.form.get("notebook", DEFAULT_NOTEBOOK).strip() or DEFAULT_NOTEBOOK
    chunk_size = request.form.get("chunk_size", type=int)
    chunk_overlap = request.form.get("chunk_overlap", type=int)

    results = []
    for uploaded_file in files:
        if uploaded_file.filename == "":
            continue
        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": uploaded_file.filename, "status": "error",
                             "message": "Unsupported file type."})
            continue

        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            uploaded_file.save(file_path)
            text = extract_text(file_path)
            if not text.strip():
                results.append({"filename": filename, "status": "error",
                                 "message": "No extractable text found."})
                continue

            chunk_count = ingest_document(
                text=text, source=filename, collection_name=notebook,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
            results.append({"filename": filename, "status": "success", "chunk_count": chunk_count})

        except Exception as e:
            results.append({"filename": uploaded_file.filename, "status": "error", "message": str(e)})
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return jsonify({"status": "success", "results": results,
                     "sources": vectorstore.list_sources(notebook)})


@app.route("/api/sources/<path:source_name>", methods=["DELETE"])
def delete_source(source_name):
    notebook = request.args.get("notebook", DEFAULT_NOTEBOOK)
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

    notebook = data.get("notebook", DEFAULT_NOTEBOOK).strip() or DEFAULT_NOTEBOOK
    top_k = int(data.get("top_k", DEFAULT_TOP_K))
    alpha = float(data.get("alpha", DEFAULT_HYBRID_ALPHA))

    try:
        result = answer_query(notebook, query_text, top_k=top_k, alpha=alpha)
        return jsonify({
            "status": "success",
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/chat/history", methods=["GET"])
def get_chat_history():
    notebook = request.args.get("notebook", DEFAULT_NOTEBOOK)
    return jsonify({"status": "success", "history": chat_history.get_history(notebook)})


@app.route("/api/chat/history", methods=["DELETE"])
def reset_chat_history():
    notebook = request.args.get("notebook", DEFAULT_NOTEBOOK)
    chat_history.clear_history(notebook)
    return jsonify({"status": "success", "message": "Conversation cleared."})


# ----------------------------------------------------------------
# Notebooks (collections)
# ----------------------------------------------------------------

@app.route("/api/notebooks", methods=["GET"])
def list_notebooks():
    return jsonify({"status": "success", "notebooks": vectorstore.list_collections()})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", host="0.0.0.0", port=port)
