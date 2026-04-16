#!/usr/bin/env python3
"""
main.py  –  CLI entry-point for the Self-Correcting RAG Learning Assistant
===========================================================================

Usage
-----
Original RAG Q&A mode:
    python main.py                 # interactive CLI loop
    python main.py --rebuild       # force-rebuild the vector store, then loop
    python main.py --mock          # use MockCritic (no API key needed)

Learning path pipeline:
    python main.py --load-path "ML Engineer.json"   # load + validate a path
    python main.py --sessions                       # list sessions with status
    python main.py --study 3                        # study session 3 (interactive)
    python main.py --quiz 3                         # take quiz for session 3
    python main.py --mastery                        # view mastery summary
    python main.py --adapt                          # generate adapted path
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

import json

import config
from ingestion import ingest
from vector_store import build_vectorstore, load_vectorstore
from critic import GeminiCritic, GemmaCritic, MockCritic, create_critic
from pipeline import RAGPipeline, QueryResult
from database import (
    create_learning_path, get_learning_path, get_user_learning_paths,
    create_session_progress, get_all_session_progress, update_session_progress,
    get_user_by_username,
)
from path_validator import load_learning_path, validate_and_correct_path
from session_orchestrator import ingest_session, session_id_for
from question_generator import generate_session_quiz
from answer_evaluator import grade_quiz
from mastery_tracker import get_mastery_summary
from path_adapter import adapt_path


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


# ── CLI state (tracks active path for CLI session) ───────────────────
_CLI_USER_ID = 1       # CLI uses a default user; override with --user if needed
_CLI_PATH_FILE = config.BASE_DIR / ".active_path"


def _save_active_path_id(path_id: int) -> None:
    _CLI_PATH_FILE.write_text(str(path_id))


def _load_active_path_id() -> int | None:
    if _CLI_PATH_FILE.exists():
        try:
            return int(_CLI_PATH_FILE.read_text().strip())
        except ValueError:
            return None
    return None


def _ensure_cli_user() -> int:
    """Ensure a CLI user exists in the database."""
    from database import create_user, get_user_by_username
    from auth import hash_password
    user = get_user_by_username("cli_user")
    if user:
        return user["id"]
    return create_user("cli_user", "cli@localhost", hash_password("cli_default"))


# ── learning pipeline commands ───────────────────────────────────────

def _run_learning_pipeline(args) -> None:
    """Dispatch learning pipeline CLI commands."""
    user_id = _ensure_cli_user()

    if args.mock:
        critic = MockCritic(fixed_confidence=90.0)
    else:
        critic = create_critic()

    if args.load_path:
        _cmd_load_path(args.load_path, user_id)
    elif args.sessions:
        _cmd_list_sessions(user_id)
    elif args.study is not None:
        _cmd_study(args.study, user_id, critic)
    elif args.quiz is not None:
        _cmd_quiz(args.quiz, user_id, critic)
    elif args.mastery:
        _cmd_mastery(user_id)
    elif args.adapt:
        _cmd_adapt(user_id, critic)


def _cmd_load_path(filename: str, user_id: int) -> None:
    """Load and validate a learning path."""
    filepath = config.LEARNING_PATH_DIR / filename
    if not filepath.exists():
        print(f"[!] File not found: {filepath}")
        return

    print(f"[*] Loading learning path: {filename}")
    loaded = load_learning_path(filepath)
    career_goal = loaded["career_goal"]
    path_data = loaded["path_data"]
    sessions = path_data.get("learning_path", [])
    print(f"[*] Career goal: {career_goal}")
    print(f"[*] Sessions: {len(sessions)}")

    print(f"\n[*] Validating and correcting path...")
    corrected_data, change_log, report = validate_and_correct_path(filepath)

    # Save to database
    path_id = create_learning_path(
        user_id=user_id,
        filename=filename,
        career_goal=career_goal,
        original_path=json.dumps(path_data, ensure_ascii=False),
    )
    from database import update_learning_path
    update_learning_path(
        path_id,
        corrected_path=json.dumps(corrected_data, ensure_ascii=False),
        status="validated",
    )

    # Create session records
    for session in corrected_data.get("learning_path", []):
        create_session_progress(
            path_id=path_id,
            session_number=session.get("session_number", 0),
            title=session.get("title", ""),
        )

    _save_active_path_id(path_id)

    print(f"\n{change_log}")
    print(f"\n[*] Path loaded as ID {path_id} (set as active).")


def _cmd_list_sessions(user_id: int) -> None:
    """List sessions for the active path."""
    path_id = _load_active_path_id()
    if not path_id:
        print("[!] No active path. Load one with --load-path first.")
        return

    path_record = get_learning_path(path_id)
    if not path_record:
        print(f"[!] Path ID {path_id} not found.")
        return

    print(f"\nLearning Path: {path_record['career_goal']} (ID: {path_id})")
    print("=" * 60)

    progress = get_all_session_progress(path_id)
    for p in progress:
        status_icon = {
            "not_started": "[ ]",
            "in_progress": "[~]",
            "completed": "[x]",
            "skipped": "[-]",
        }.get(p["status"], "[?]")
        vs = " (indexed)" if p["vectorstore_built"] else ""
        print(f"  {status_icon} Session {p['session_number']}: {p['title']}{vs}")


def _cmd_study(session_number: int, user_id: int, critic) -> None:
    """Interactive study mode for a session."""
    path_id = _load_active_path_id()
    if not path_id:
        print("[!] No active path. Load one with --load-path first.")
        return

    path_record = get_learning_path(path_id)
    path_data = json.loads(path_record.get("corrected_path") or path_record["original_path"])
    target = None
    for s in path_data.get("learning_path", []):
        if s.get("session_number") == session_number:
            target = s
            break
    if not target:
        print(f"[!] Session {session_number} not found.")
        return

    print(f"\n[*] Starting session {session_number}: {target.get('title', '')}")
    print("[*] Ingesting session content...")

    sid = ingest_session(
        session=target,
        path_id=str(path_id),
        career_goal=path_record["career_goal"],
    )

    from datetime import datetime, timezone
    update_session_progress(
        path_id, session_number,
        status="in_progress",
        vectorstore_built=1,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    # Create a pipeline for session queries
    from unittest.mock import MagicMock
    pipeline = RAGPipeline(vectorstore=MagicMock(), critic=critic)

    print(f"[*] Session ready. Ask questions about the material.")
    print(f"[*] Type 'done' to finish studying, 'quit' to exit.\n")

    while True:
        try:
            question = input(">> Study Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"done", "finish"}:
            update_session_progress(
                path_id, session_number,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            print("[*] Session marked as completed.")
            break
        if question.lower() in {"quit", "exit", "q"}:
            break

        result = pipeline.session_query(sid, question)
        _print_result(result)


def _cmd_quiz(session_number: int, user_id: int, critic) -> None:
    """Take a quiz for a session."""
    path_id = _load_active_path_id()
    if not path_id:
        print("[!] No active path. Load one with --load-path first.")
        return

    path_record = get_learning_path(path_id)
    path_data = json.loads(path_record.get("corrected_path") or path_record["original_path"])
    target = None
    for s in path_data.get("learning_path", []):
        if s.get("session_number") == session_number:
            target = s
            break
    if not target:
        print(f"[!] Session {session_number} not found.")
        return

    sid = session_id_for(str(path_id), session_number)

    print(f"\n[*] Generating quiz for session {session_number}: {target.get('title', '')}...")
    quiz = generate_session_quiz(sid, target, critic)

    if not quiz.questions:
        print("[!] Could not generate any questions.")
        return

    print(f"[*] Quiz: {len(quiz.questions)} questions "
          f"({quiz.total_mcq} MCQ, {quiz.total_open} open-ended)\n")

    answers = []
    for i, q in enumerate(quiz.questions, 1):
        source_label = "[Materials]" if q.source == "your_materials" else "[General]"
        print(f"Q{i} {source_label} ({q.difficulty}): {q.question_text}")
        if q.question_type == "mcq" and q.options:
            for key, val in q.options.items():
                print(f"    {key}) {val}")
            answer = input("  Your answer (A/B/C/D): ").strip()
        else:
            answer = input("  Your answer: ").strip()

        # Allow doubt clarification mid-quiz
        if answer.lower().startswith("?"):
            doubt = answer[1:].strip()
            if doubt:
                from unittest.mock import MagicMock
                pipeline = RAGPipeline(vectorstore=MagicMock(), critic=critic)
                try:
                    doubt_result = pipeline.session_query(sid, doubt)
                    print(f"\n  [Clarification]: {doubt_result.answer[:300]}\n")
                except Exception:
                    print("  [Clarification not available for this session]\n")
                answer = input("  Now your answer: ").strip()

        answers.append(answer)
        print()

    print("[*] Grading quiz...")
    result = grade_quiz(
        quiz=quiz,
        user_answers=answers,
        critic=critic,
        user_id=user_id,
        path_id=path_id,
    )

    print(f"\n{'=' * 60}")
    print(f"Score: {result.percentage:.1f}% ({result.total_score:.0f}/{result.max_score:.0f})")
    print(f"{'=' * 60}")

    for i, a in enumerate(result.answers, 1):
        icon = "+" if a.score >= 50 else "-"
        print(f"  [{icon}] Q{i}: {a.score:.0f}/100 — {a.feedback[:80]}")

    if result.per_skill_scores:
        print(f"\nPer-skill mastery:")
        for skill, score in result.per_skill_scores.items():
            print(f"  {skill}: {score:.1f}%")


def _cmd_mastery(user_id: int) -> None:
    """View mastery summary."""
    path_id = _load_active_path_id()
    if not path_id:
        print("[!] No active path. Load one with --load-path first.")
        return

    summary = get_mastery_summary(user_id, path_id)

    print(f"\nMastery Summary (Path ID: {path_id})")
    print("=" * 50)
    print(f"  Total skills assessed: {summary.total_skills}")
    print(f"  Mastered (>={config.MASTERY_THRESHOLD_SKIP}%): {summary.mastered_count}")
    print(f"  Needs review: {summary.review_count}")
    print(f"  Weak (<{config.MASTERY_THRESHOLD_REVIEW}%): {summary.weak_count}")
    print(f"  Not assessed: {summary.not_assessed_count}")
    print(f"  Overall: {summary.overall_percentage:.1f}%")

    if summary.skills:
        print(f"\nSkill Breakdown:")
        for s in summary.skills:
            icon = {"mastered": "+", "review": "~", "weak": "-", "not_assessed": "?"}.get(s.status, "?")
            print(f"  [{icon}] {s.skill_label}: {s.mastery_score:.1f}% ({s.attempts_count} attempts)")


def _cmd_adapt(user_id: int, critic) -> None:
    """Generate an adapted learning path."""
    path_id = _load_active_path_id()
    if not path_id:
        print("[!] No active path. Load one with --load-path first.")
        return

    path_record = get_learning_path(path_id)
    corrected_path = json.loads(
        path_record.get("corrected_path") or path_record["original_path"]
    )
    corrected_path["_filename"] = Path(path_record["filename"]).stem

    print("[*] Adapting learning path based on mastery scores...")
    adapted_data, adaptation_log = adapt_path(
        user_id=user_id,
        path_id=path_id,
        corrected_path=corrected_path,
        career_goal=path_record["career_goal"],
    )

    print(f"\n{adaptation_log}")
    print(f"\n[*] Adapted path saved to {config.CORRECTED_PATH_DIR}/")


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
    # Learning path pipeline arguments
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Load and validate a learning path JSON file (e.g. 'ML Engineer.json').",
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="List sessions and their progress for the active learning path.",
    )
    parser.add_argument(
        "--study",
        type=int,
        default=None,
        metavar="N",
        help="Enter interactive study mode for session N.",
    )
    parser.add_argument(
        "--quiz",
        type=int,
        default=None,
        metavar="N",
        help="Take a quiz for session N.",
    )
    parser.add_argument(
        "--mastery",
        action="store_true",
        help="View mastery summary for the active learning path.",
    )
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Generate an adapted learning path based on mastery scores.",
    )
    args = parser.parse_args()

    _setup_logging()

    # ---- Dispatch to learning pipeline commands -----------------------
    is_learning_cmd = any([
        args.load_path, args.sessions, args.study is not None,
        args.quiz is not None, args.mastery, args.adapt,
    ])

    if is_learning_cmd:
        _run_learning_pipeline(args)
        return

    # ---- Original RAG Q&A mode ---------------------------------------
    vectorstore = _get_or_build_vectorstore(force_rebuild=args.rebuild)

    if args.mock:
        critic = MockCritic(fixed_confidence=90.0)
        print("[*] Using MockCritic (offline mode).\n")
    else:
        critic = create_critic()
        print(f"[*] Using {type(critic).__name__} (backend={config.CRITIC_BACKEND}).\n")

    pipeline = RAGPipeline(
        vectorstore=vectorstore,
        critic=critic,
        similarity_threshold=args.threshold,
        confidence_threshold=args.confidence,
    )

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
