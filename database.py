"""
database.py – SQLite persistence for the RAG Learning Assistant
----------------------------------------------------------------
Lightweight persistence using built-in sqlite3.
Creates the DB file on first import if it doesn't exist.
Covers: users, chat history, learning paths, session progress,
quiz attempts/answers, and skill mastery tracking.
"""

from __future__ import annotations

import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / "users.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                email       TEXT    UNIQUE NOT NULL,
                password    TEXT    NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                question    TEXT    NOT NULL,
                answer      TEXT    NOT NULL,
                source      TEXT    NOT NULL,
                metadata    TEXT    DEFAULT '{}',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        # ── Learning Path Pipeline Tables ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_paths (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL,
                filename        TEXT    NOT NULL,
                career_goal     TEXT    NOT NULL,
                original_path   TEXT    NOT NULL,
                corrected_path  TEXT,
                status          TEXT    NOT NULL DEFAULT 'loaded',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_progress (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                path_id           INTEGER NOT NULL,
                session_number    INTEGER NOT NULL,
                title             TEXT    NOT NULL,
                status            TEXT    NOT NULL DEFAULT 'not_started',
                vectorstore_built INTEGER NOT NULL DEFAULT 0,
                started_at        TIMESTAMP,
                completed_at      TIMESTAMP,
                FOREIGN KEY (path_id) REFERENCES learning_paths(id),
                UNIQUE(path_id, session_number)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                path_id         INTEGER NOT NULL,
                session_number  INTEGER NOT NULL,
                user_id         INTEGER NOT NULL,
                attempt_number  INTEGER NOT NULL,
                total_score     REAL    NOT NULL DEFAULT 0,
                max_score       REAL    NOT NULL DEFAULT 0,
                percentage      REAL    NOT NULL DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (path_id) REFERENCES learning_paths(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_answers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id      INTEGER NOT NULL,
                question_text   TEXT    NOT NULL,
                question_type   TEXT    NOT NULL,
                question_source TEXT    NOT NULL DEFAULT 'your_materials',
                user_answer     TEXT,
                correct_answer  TEXT,
                score           REAL    NOT NULL DEFAULT 0,
                max_score       REAL    NOT NULL DEFAULT 100,
                feedback        TEXT,
                FOREIGN KEY (attempt_id) REFERENCES quiz_attempts(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_mastery (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL,
                path_id         INTEGER NOT NULL,
                skill_label     TEXT    NOT NULL,
                session_number  INTEGER NOT NULL,
                mastery_score   REAL    NOT NULL DEFAULT 0,
                attempts_count  INTEGER NOT NULL DEFAULT 0,
                last_assessed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (path_id) REFERENCES learning_paths(id),
                UNIQUE(user_id, path_id, skill_label)
            )
        """)

        # ── Schema migrations (idempotent ALTER TABLE blocks) ──
        _migrations = [
            "ALTER TABLE quiz_attempts ADD COLUMN self_confidence REAL",
            "ALTER TABLE session_progress ADD COLUMN doubt_query_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE session_progress ADD COLUMN study_duration_seconds INTEGER NOT NULL DEFAULT 0",
        ]
        for stmt in _migrations:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists


def create_user(username: str, email: str, hashed_password: str) -> int:
    """Insert a new user; returns their ID."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def get_user_by_username(username: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def save_chat(user_id: int, question: str, answer: str, source: str, metadata: str = "{}") -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, question, answer, source, metadata) VALUES (?, ?, ?, ?, ?)",
            (user_id, question, answer, source, metadata),
        )


def get_chat_history(user_id: int, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT question, answer, source, metadata, created_at FROM chat_history "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# Learning Paths
# ──────────────────────────────────────────────

def create_learning_path(user_id: int, filename: str, career_goal: str,
                         original_path: str) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO learning_paths (user_id, filename, career_goal, original_path) "
            "VALUES (?, ?, ?, ?)",
            (user_id, filename, career_goal, original_path),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def get_learning_path(path_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM learning_paths WHERE id = ?", (path_id,)
        ).fetchone()
        return dict(row) if row else None


def get_user_learning_paths(user_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM learning_paths WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_learning_path(path_id: int, **kwargs) -> None:
    allowed = {"corrected_path", "status"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [path_id]
    with get_db() as conn:
        conn.execute(
            f"UPDATE learning_paths SET {set_clause} WHERE id = ?", values
        )


# ──────────────────────────────────────────────
# Session Progress
# ──────────────────────────────────────────────

def create_session_progress(path_id: int, session_number: int, title: str) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO session_progress (path_id, session_number, title) "
            "VALUES (?, ?, ?)",
            (path_id, session_number, title),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def get_session_progress(path_id: int, session_number: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM session_progress WHERE path_id = ? AND session_number = ?",
            (path_id, session_number),
        ).fetchone()
        return dict(row) if row else None


def get_all_session_progress(path_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM session_progress WHERE path_id = ? ORDER BY session_number",
            (path_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_session_progress(path_id: int, session_number: int, **kwargs) -> None:
    allowed = {"status", "vectorstore_built", "started_at", "completed_at",
               "doubt_query_count", "study_duration_seconds"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [path_id, session_number]
    with get_db() as conn:
        conn.execute(
            f"UPDATE session_progress SET {set_clause} "
            "WHERE path_id = ? AND session_number = ?",
            values,
        )


# ──────────────────────────────────────────────
# Quiz Attempts & Answers
# ──────────────────────────────────────────────

def create_quiz_attempt(path_id: int, session_number: int, user_id: int,
                        attempt_number: int,
                        self_confidence: float | None = None) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO quiz_attempts "
            "(path_id, session_number, user_id, attempt_number, self_confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (path_id, session_number, user_id, attempt_number, self_confidence),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def increment_doubt_count(path_id: int, session_number: int) -> None:
    """Increment the doubt-query counter for a session (cognitive-load signal)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE session_progress "
            "SET doubt_query_count = doubt_query_count + 1 "
            "WHERE path_id = ? AND session_number = ?",
            (path_id, session_number),
        )


def get_calibration_data(user_id: int, path_id: int) -> list[dict]:
    """Return (self_confidence, percentage) pairs across all quiz attempts
    for the user+path — used for Dunning-Kruger / calibration analysis."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT session_number, attempt_number, self_confidence, percentage, created_at "
            "FROM quiz_attempts "
            "WHERE user_id = ? AND path_id = ? AND self_confidence IS NOT NULL "
            "ORDER BY created_at",
            (user_id, path_id),
        ).fetchall()
        return [dict(r) for r in rows]


def get_cognitive_load_data(path_id: int) -> list[dict]:
    """Return per-session cognitive-load signals (doubt count + duration)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT session_number, title, doubt_query_count, "
            "study_duration_seconds, status "
            "FROM session_progress WHERE path_id = ? ORDER BY session_number",
            (path_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_quiz_attempt(attempt_id: int, total_score: float,
                        max_score: float, percentage: float) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE quiz_attempts SET total_score = ?, max_score = ?, percentage = ? "
            "WHERE id = ?",
            (total_score, max_score, percentage, attempt_id),
        )


def get_quiz_attempts(path_id: int, session_number: int,
                      user_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM quiz_attempts "
            "WHERE path_id = ? AND session_number = ? AND user_id = ? "
            "ORDER BY attempt_number DESC",
            (path_id, session_number, user_id),
        ).fetchall()
        return [dict(r) for r in rows]


def save_quiz_answer(attempt_id: int, question_text: str, question_type: str,
                     question_source: str, user_answer: str | None,
                     correct_answer: str | None, score: float,
                     max_score: float, feedback: str | None) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO quiz_answers "
            "(attempt_id, question_text, question_type, question_source, "
            "user_answer, correct_answer, score, max_score, feedback) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (attempt_id, question_text, question_type, question_source,
             user_answer, correct_answer, score, max_score, feedback),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def get_quiz_answers(attempt_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM quiz_answers WHERE attempt_id = ? ORDER BY id",
            (attempt_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# Skill Mastery
# ──────────────────────────────────────────────

def upsert_skill_mastery(user_id: int, path_id: int, skill_label: str,
                         session_number: int, new_score: float) -> None:
    """Update mastery with weighted running average (70% new, 30% old)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT mastery_score, attempts_count FROM skill_mastery "
            "WHERE user_id = ? AND path_id = ? AND skill_label = ?",
            (user_id, path_id, skill_label),
        ).fetchone()
        if row:
            old_score = row["mastery_score"]
            count = row["attempts_count"]
            updated_score = 0.7 * new_score + 0.3 * old_score
            conn.execute(
                "UPDATE skill_mastery SET mastery_score = ?, attempts_count = ?, "
                "last_assessed_at = CURRENT_TIMESTAMP "
                "WHERE user_id = ? AND path_id = ? AND skill_label = ?",
                (updated_score, count + 1, user_id, path_id, skill_label),
            )
        else:
            conn.execute(
                "INSERT INTO skill_mastery "
                "(user_id, path_id, skill_label, session_number, "
                "mastery_score, attempts_count, last_assessed_at) "
                "VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)",
                (user_id, path_id, skill_label, session_number, new_score),
            )


def get_skill_mastery(user_id: int, path_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM skill_mastery "
            "WHERE user_id = ? AND path_id = ? ORDER BY session_number, skill_label",
            (user_id, path_id),
        ).fetchall()
        return [dict(r) for r in rows]


def get_skill_mastery_by_session(user_id: int, path_id: int,
                                 session_number: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM skill_mastery "
            "WHERE user_id = ? AND path_id = ? AND session_number = ?",
            (user_id, path_id, session_number),
        ).fetchall()
        return [dict(r) for r in rows]


# Auto-initialise on import
init_db()
