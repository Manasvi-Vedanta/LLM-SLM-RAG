#!/usr/bin/env python3
"""
main.py  –  CLI entry-point for the Self-Correcting RAG System
===============================================================

Usage
-----
    python main.py                 # interactive CLI loop
    python main.py --rebuild       # force-rebuild the vector store, then loop
    python main.py --mock          # use MockCritic (no API key needed)

The first run will automatically ingest the PDFs and build the FAISS index.
Subsequent runs load the persisted index from disk unless ``--rebuild`` is
passed.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import textwrap
from pathlib import Path

# Force UTF-8 stdout so box-drawing / emoji chars don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import config
from ingestion import ingest
from vector_store import build_vectorstore, load_vectorstore
from critic import GeminiCritic, MockCritic
from pipeline import RAGPipeline, QueryResult


# ── helpers ───────────────────────────────────────────────────────────
def _setup_logging(level: str = config.LOG_LEVEL) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_or_build_vectorstore(force_rebuild: bool = False):
    """Return a FAISS vectorstore, building it from PDFs if necessary."""
    index_path = config.VECTORSTORE_DIR / "index.faiss"
    if not force_rebuild and index_path.exists():
        print("[*] Loading existing FAISS index …")
        return load_vectorstore()
    else:
        print("[*] Building FAISS index from PDFs (this may take a minute) …")
        chunks = ingest()
        print(f"[*] Ingested {len(chunks)} chunks from Dataset/")
        vs = build_vectorstore(chunks)
        print("[*] FAISS index built and persisted.\n")
        return vs


def _print_result(result: QueryResult) -> None:
    """Pretty-print a QueryResult to the terminal."""
    width = min(90, 90)
    print("\n" + "=" * width)

    if result.source == "out_of_scope":
        print("[OUT OF SCOPE]")
        if result.retrieval and result.retrieval.best_score is not None:
            print(f"    Best similarity score : {result.retrieval.best_score:.4f}")
            print(f"    Threshold             : {result.metadata.get('threshold', '?')}")
    elif result.source == "document":
        print("[ANSWER FROM DOCUMENT]")
        print(f"    Source file : {result.metadata.get('source_file', '?')}")
        print(f"    Page        : {result.metadata.get('page', '?')}")
        print(f"    Similarity  : {result.metadata.get('similarity_score', 0):.4f}")
        print(f"    Confidence  : {result.metadata.get('confidence', 0):.1f}%")
    else:
        print("[ANSWER FROM GENERAL KNOWLEDGE]  (document was insufficient)")
        print(f"    Similarity  : {result.metadata.get('similarity_score', 0):.4f}")
        print(f"    Confidence  : {result.metadata.get('confidence', 0):.1f}%")

    print("-" * width)
    # Wrap the answer text for readability
    for line in result.answer.splitlines():
        print(textwrap.fill(line, width=width) if line.strip() else "")
    print("=" * width + "\n")


# ── main loop ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Correcting RAG System (Actor-Critic)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild the FAISS vector store from PDFs.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockCritic instead of Gemini (no API key required).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=f"Override the confidence threshold (default {config.CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Override the similarity threshold (default {config.SIMILARITY_THRESHOLD}).",
    )
    args = parser.parse_args()

    _setup_logging()

    # ---- Vector store (Actor) -----------------------------------------
    vectorstore = _get_or_build_vectorstore(force_rebuild=args.rebuild)

    # ---- Critic -------------------------------------------------------
    if args.mock:
        critic = MockCritic(fixed_confidence=90.0)
        print("[*] Using MockCritic (offline mode).\n")
    else:
        if config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            print(
                "⚠️  No GEMINI_API_KEY set.  Falling back to MockCritic.\n"
                "    Set the env-var or edit config.py to use the real API.\n"
            )
            critic = MockCritic(fixed_confidence=90.0)
        else:
            critic = GeminiCritic()
            print(f"[*] Using GeminiCritic (model={config.GEMINI_MODEL_NAME}).\n")

    # ---- Pipeline -----------------------------------------------------
    pipeline = RAGPipeline(
        vectorstore=vectorstore,
        critic=critic,
        similarity_threshold=args.threshold,
        confidence_threshold=args.confidence,
    )

    # ---- Interactive loop ---------------------------------------------
    print("+" + "=" * 62 + "+")
    print("|   Self-Correcting RAG System  -  Actor-Critic Architecture   |")
    print("|   Type your question and press Enter.  Type 'quit' to exit.  |")
    print("+" + "=" * 62 + "+\n")

    while True:
        try:
            question = input(">> You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        result = pipeline.query(question)
        _print_result(result)


if __name__ == "__main__":
    main()
