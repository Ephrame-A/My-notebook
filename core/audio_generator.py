"""
Audio overview generator
========================
Generates a podcast-style or narrator-style audio overview of a notebook's
content.  Two phases: (1) the LLM writes a script from the stored chunks,
(2) Edge-TTS synthesises the script to MP3.
"""

import os
import asyncio
import tempfile
from typing import List

import edge_tts
from pydub import AudioSegment

from core import vectorstore
from core.rag_pipeline import _call_llm
from core.config import (
    AUDIO_VOICE_HOST_A,
    AUDIO_VOICE_HOST_B,
    AUDIO_VOICE_NARRATOR,
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_PODCAST_SYSTEM_PROMPT = """\
You are writing a script for a two-host educational podcast episode.
You will be given the full text content of a document collection.
Write a natural, engaging conversation between two hosts (HOST_A and HOST_B) \
that covers the key ideas, themes, and insights in the material. \
Focus the discussion on the user's topic/question provided below.

STRICT FORMAT — every single line must follow one of these two patterns exactly:
HOST_A: <spoken dialogue>
HOST_B: <spoken dialogue>

Rules:
- No blank lines between lines of dialogue.
- No stage directions, headers, or any other text outside those two patterns.
- The hosts build on each other's points, ask clarifying questions, \
and keep the listener engaged.
- Aim for roughly 1500-2000 words of dialogue.
- Use natural spoken language — no bullet points, no markdown."""

_NARRATOR_SYSTEM_PROMPT = """\
You are writing a script for an educational audio summary read by a single \
narrator.  You will be given the full text content of a document collection.
Write a clear, engaging monologue that covers the key ideas and themes. \
Focus on the user's topic/question provided below.

Rules:
- Write continuous prose as if speaking directly to a listener.
- Do NOT use bullet points, headers, numbered lists, or any markdown.
- Use natural spoken transitions ("First,", "What's interesting here is…", \
"To wrap up,").
- Aim for roughly 700-1000 words."""

# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

def _build_content_block(collection_name: str) -> str:
    data = vectorstore.get_all_chunks(collection_name)
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    if not docs:
        return ""
    by_source: dict = {}
    for text, meta in zip(docs, metas):
        src = (meta or {}).get("source", "unknown")
        by_source.setdefault(src, []).append(text)
    parts = []
    for src, chunks in by_source.items():
        parts.append(f"=== Source: {src} ===")
        parts.extend(chunks)
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Script generation (LLM)
# ---------------------------------------------------------------------------

def generate_script(collection_name: str, mode: str, topic: str = "",
                    api_key: str = None) -> str:
    if mode not in ("podcast", "narrator"):
        raise ValueError(f"Unknown mode '{mode}'. Choose 'podcast' or 'narrator'.")
    content_block = _build_content_block(collection_name)
    if not content_block:
        raise ValueError(
            "This notebook has no content. "
            "Add at least one source document before generating audio."
        )
    system = _PODCAST_SYSTEM_PROMPT if mode == "podcast" else _NARRATOR_SYSTEM_PROMPT
    prompt = f"{system}\n\nDocument content:\n\n{content_block}"
    if topic:
        prompt += f"\n\nUser's topic/question: {topic}"
    try:
        return _call_llm(prompt, api_key=api_key)
    except RuntimeError:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "api key" in msg or "api_key" in msg or "authenticate" in msg or "permission" in msg:
            raise RuntimeError(
                "Invalid or expired Gemini API key. Please check your key and try again."
            ) from e
        if "quota" in msg or "rate" in msg or "limit" in msg or "429" in msg:
            raise RuntimeError(
                "Gemini API rate limit reached. Please wait a moment and try again."
            ) from e
        if "safety" in msg or "blocked" in msg:
            raise RuntimeError(
                "The content was blocked by Gemini's safety filters. "
                "Try with different source documents."
            ) from e
        raise RuntimeError(
            "Failed to generate the script. Please check your API key and try again."
        ) from e

# ---------------------------------------------------------------------------
# TTS synthesis (Edge-TTS) — all segments synthesised concurrently
# ---------------------------------------------------------------------------

async def _synthesise_segment(text: str, voice: str, dest: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(dest)


async def _synthesise_all(segments: List[tuple]) -> None:
    """Synthesise all (text, voice, dest) segments concurrently."""
    await asyncio.gather(*[
        _synthesise_segment(text, voice, dest)
        for text, voice, dest in segments
    ])


def _concat_mp3s(paths: List[str], output_path: str) -> None:
    combined = AudioSegment.empty()
    for p in paths:
        combined += AudioSegment.from_mp3(p)
    combined.export(output_path, format="mp3")


def synthesise_podcast(script: str, output_path: str) -> None:
    voice_map = {"HOST_A": AUDIO_VOICE_HOST_A, "HOST_B": AUDIO_VOICE_HOST_B}
    lines = [line.strip() for line in script.splitlines() if line.strip()]

    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, line in enumerate(lines):
            for host, voice in voice_map.items():
                prefix = f"{host}:"
                if line.startswith(prefix):
                    text = line[len(prefix):].strip()
                    if not text:
                        continue
                    seg_path = os.path.join(tmp, f"seg_{i:04d}.mp3")
                    segments.append((text, voice, seg_path))
                    break

        if not segments:
            raise ValueError(
                "Script produced no synthesisable lines. "
                "The LLM may not have followed the HOST_A:/HOST_B: format."
            )

        asyncio.run(_synthesise_all(segments))
        _concat_mp3s([s[2] for s in segments], output_path)


def synthesise_narrator(script: str, output_path: str) -> None:
    text = script.strip()
    if not text:
        raise ValueError("Narrator script is empty.")
    asyncio.run(_synthesise_segment(text, AUDIO_VOICE_NARRATOR, output_path))

# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_audio_overview(
    collection_name: str,
    mode: str,
    output_path: str,
    topic: str = "",
    api_key: str = None,
) -> str:
    """Generates a script and synthesises it to MP3. Returns the script text."""
    script = generate_script(collection_name, mode, topic=topic, api_key=api_key)
    if mode == "podcast":
        synthesise_podcast(script, output_path)
    else:
        synthesise_narrator(script, output_path)
    return script
