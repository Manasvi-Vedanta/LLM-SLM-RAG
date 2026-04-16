"""
server.py – FastAPI backend for the Self-Correcting RAG Learning Assistant
===========================================================================

Endpoints
~~~~~~~~~
Auth:
  POST /api/auth/signup      – register a new user
  POST /api/auth/login       – authenticate and receive JWT
  GET  /api/auth/me          – get current user info

Original RAG chat:
  POST /api/chat             – send a question to the RAG pipeline
  GET  /api/chat/history     – retrieve chat history

Learning path pipeline:
  POST /api/path/load                    – load a learning path JSON
  POST /api/path/{path_id}/validate      – validate & correct path
  GET  /api/path/{path_id}/sessions      – list sessions with progress
  POST /api/session/{session_id}/start   – start a session (lazy ingestion)
  POST /api/session/{session_id}/ask     – doubt clarification
  POST /api/session/{session_id}/complete – mark session done
  POST /api/quiz/{session_id}/generate   – generate quiz
  POST /api/quiz/{session_id}/submit     – submit quiz answers
  GET  /api/quiz/{session_id}/results    – past quiz results
  GET  /api/mastery/{path_id}            – mastery scores
  POST /api/path/{path_id}/adapt         – trigger path adaptation

Pages:
  GET  /                     – landing page
  GET  /login, /signup, /chat, /learn – UI pages

Run
~~~
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import sys

# Force UTF-8 for Windows console compatibility
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from pathlib import Path

import config
from auth import hash_password, verify_password, create_access_token, decode_access_token
from database import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_user_by_id,
    save_chat,
    get_chat_history,
    create_learning_path,
    get_learning_path,
    get_user_learning_paths,
    update_learning_path,
    create_session_progress,
    get_session_progress,
    get_all_session_progress,
    update_session_progress,
    get_quiz_attempts,
    get_quiz_answers,
)
from ingestion import ingest
from vector_store import build_vectorstore, load_vectorstore
from critic import GeminiCritic, GemmaCritic, MockCritic, create_critic
from pipeline import RAGPipeline
from path_validator import load_learning_path, validate_and_correct_path
from session_orchestrator import ingest_session, session_id_for, get_session_materials
from question_generator import generate_session_quiz
from answer_evaluator import grade_quiz
from mastery_tracker import get_mastery_summary, get_session_readiness
from path_adapter import adapt_path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="Self-Correcting RAG System",
    description="Actor-Critic RAG with web UI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Security scheme ──────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


# ── Startup: load RAG pipeline once ──────────────────────────────────
_pipeline: RAGPipeline | None = None
_critic = None  # Critic instance, shared across endpoints


@app.on_event("startup")
async def startup_event():
    global _pipeline, _critic
    logger.info("Loading RAG pipeline...")

    # Initialise critic (reads CRITIC_BACKEND from config.py)
    _critic = create_critic()
    logger.info(
        "Critic ready: %s (backend=%s)",
        type(_critic).__name__,
        config.CRITIC_BACKEND,
    )

    # Build or load global vector store (optional — may not have PDFs)
    index_path = config.VECTORSTORE_DIR / "index.faiss"
    try:
        if index_path.exists():
            logger.info("Loading existing FAISS index...")
            vectorstore = load_vectorstore()
        else:
            logger.info("Building FAISS index from PDFs...")
            chunks = ingest()
            logger.info("Ingested %d chunks", len(chunks))
            vectorstore = build_vectorstore(chunks)

        _pipeline = RAGPipeline(vectorstore=vectorstore, critic=_critic)
        logger.info("RAG pipeline ready (global vectorstore loaded).")
    except FileNotFoundError:
        logger.warning(
            "No PDFs found in %s — global RAG pipeline not available. "
            "Learning path pipeline is still fully functional.",
            config.DATASET_DIR,
        )
        _pipeline = None


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return _pipeline


# ── Auth dependency ──────────────────────────────────────────────────
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(payload.get("user_id"))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── Request / Response models ────────────────────────────────────────
class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: str
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class LoadPathRequest(BaseModel):
    filename: str = Field(..., description="JSON filename, e.g. 'ML Engineer.json'")


class SessionAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QuizSubmitRequest(BaseModel):
    answers: list[str] = Field(..., description="Answers in question order")


# ── Page routes ──────────────────────────────────────────────────────
@app.get("/")
async def landing_page():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/login")
async def login_page():
    return FileResponse(str(STATIC_DIR / "login.html"))


@app.get("/signup")
async def signup_page():
    return FileResponse(str(STATIC_DIR / "signup.html"))


@app.get("/chat")
async def chat_page():
    return FileResponse(str(STATIC_DIR / "chat.html"))


@app.get("/learn")
async def learn_page():
    return FileResponse(str(STATIC_DIR / "learn.html"))


# ── Auth API ─────────────────────────────────────────────────────────
@app.post("/api/auth/signup")
async def signup(req: SignupRequest):
    # Check duplicates
    if get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="Username already exists")
    if get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = hash_password(req.password)
    user_id = create_user(req.username, req.email, hashed)
    token = create_access_token({"user_id": user_id, "username": req.username})

    return {
        "message": "Account created successfully",
        "token": token,
        "user": {"id": user_id, "username": req.username, "email": req.email},
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"user_id": user["id"], "username": user["username"]})
    return {
        "message": "Login successful",
        "token": token,
        "user": {"id": user["id"], "username": user["username"], "email": user["email"]},
    }


@app.get("/api/auth/me")
async def me(user=Depends(get_current_user)):
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
    }


# ── Chat API ─────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    pipeline = get_pipeline()
    result = pipeline.query(req.question)

    # Build response
    response_data = {
        "answer": result.answer,
        "source": result.source,
        "metadata": result.metadata,
    }

    # Persist to history
    save_chat(
        user_id=user["id"],
        question=req.question,
        answer=result.answer,
        source=result.source,
        metadata=json.dumps(result.metadata, default=str),
    )

    return response_data


@app.get("/api/chat/history")
async def chat_history_endpoint(user=Depends(get_current_user)):
    history = get_chat_history(user["id"])
    return {"history": history}


# ── Health check ─────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


# ══════════════════════════════════════════════════════════════════════
# Learning Path Pipeline API
# ══════════════════════════════════════════════════════════════════════

def _get_critic():
    """Return the shared critic instance."""
    if _critic is None:
        raise HTTPException(status_code=503, detail="Critic not initialized")
    return _critic


def _get_learning_pipeline(path_id: int) -> RAGPipeline:
    """Create a lightweight pipeline with the shared critic (no global vectorstore needed)."""
    critic = _get_critic()
    # For session queries, session_query() loads its own vectorstore.
    # We need a RAGPipeline instance but the global vectorstore isn't required.
    if _pipeline is not None:
        return _pipeline
    # Create a minimal pipeline — session_query doesn't use self.vectorstore
    from unittest.mock import MagicMock
    return RAGPipeline(vectorstore=MagicMock(), critic=critic)


# ── Path management ──────────────────────────────────────────────────

@app.get("/api/path/available")
async def list_available_paths():
    """List all learning path JSON files available for loading."""
    path_dir = config.LEARNING_PATH_DIR
    if not path_dir.exists():
        return {"paths": []}
    files = sorted(p.name for p in path_dir.glob("*.json"))
    return {"paths": files}


@app.post("/api/path/load")
async def load_path(req: LoadPathRequest, user=Depends(get_current_user)):
    """Load a learning path JSON file and register it in the database."""
    filepath = config.LEARNING_PATH_DIR / req.filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.filename}")

    loaded = load_learning_path(filepath)
    path_data = loaded["path_data"]
    career_goal = loaded["career_goal"]

    # Save to database
    path_id = create_learning_path(
        user_id=user["id"],
        filename=req.filename,
        career_goal=career_goal,
        original_path=json.dumps(path_data, ensure_ascii=False),
    )

    # Create session progress records
    for session in path_data.get("learning_path", []):
        create_session_progress(
            path_id=path_id,
            session_number=session.get("session_number", 0),
            title=session.get("title", ""),
        )

    return {
        "path_id": path_id,
        "career_goal": career_goal,
        "total_sessions": len(path_data.get("learning_path", [])),
        "status": "loaded",
    }


@app.post("/api/path/{path_id}/validate")
async def validate_path(path_id: int, user=Depends(get_current_user)):
    """Trigger validation and correction of a loaded learning path."""
    path_record = get_learning_path(path_id)
    if not path_record:
        raise HTTPException(status_code=404, detail="Learning path not found")
    if path_record["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your learning path")

    filepath = config.LEARNING_PATH_DIR / path_record["filename"]
    corrected_data, change_log, report = validate_and_correct_path(filepath)

    # Update database
    update_learning_path(
        path_id,
        corrected_path=json.dumps(corrected_data, ensure_ascii=False),
        status="validated",
    )

    # Update session progress with corrected titles
    for session in corrected_data.get("learning_path", []):
        sn = session.get("session_number", 0)
        existing = get_session_progress(path_id, sn)
        if existing:
            update_session_progress(path_id, sn, status="not_started")

    return {
        "path_id": path_id,
        "career_goal": report.career_goal,
        "total_sessions": report.total_sessions,
        "flagged": report.flagged_count,
        "corrected": report.corrected_count,
        "change_log": change_log,
        "status": "validated",
    }


@app.get("/api/path/{path_id}/sessions")
async def list_sessions(path_id: int, user=Depends(get_current_user)):
    """List all sessions with their progress status."""
    path_record = get_learning_path(path_id)
    if not path_record:
        raise HTTPException(status_code=404, detail="Learning path not found")
    if path_record["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your learning path")

    progress = get_all_session_progress(path_id)

    # Merge with session data from the path
    path_data = json.loads(path_record.get("corrected_path") or path_record["original_path"])
    sessions_data = {s["session_number"]: s for s in path_data.get("learning_path", [])}

    sessions = []
    for p in progress:
        sn = p["session_number"]
        session_info = sessions_data.get(sn, {})
        sessions.append({
            "session_number": sn,
            "title": p["title"],
            "status": p["status"],
            "vectorstore_built": bool(p["vectorstore_built"]),
            "skills": session_info.get("skills", []),
            "difficulty_level": session_info.get("difficulty_level", ""),
            "estimated_duration_hours": session_info.get("estimated_duration_hours", 0),
            "prerequisites": session_info.get("prerequisites", []),
        })

    return {
        "path_id": path_id,
        "career_goal": path_record["career_goal"],
        "sessions": sessions,
    }


@app.get("/api/path/mine")
async def my_paths(user=Depends(get_current_user)):
    """List all learning paths for the current user."""
    paths = get_user_learning_paths(user["id"])
    return {
        "paths": [
            {
                "id": p["id"],
                "filename": p["filename"],
                "career_goal": p["career_goal"],
                "status": p["status"],
                "created_at": p["created_at"],
            }
            for p in paths
        ]
    }


# ── Session management ───────────────────────────────────────────────

@app.post("/api/session/{path_id}/{session_number}/start")
async def start_session(path_id: int, session_number: int,
                        user=Depends(get_current_user)):
    """Start studying a session — triggers lazy content ingestion."""
    path_record = get_learning_path(path_id)
    if not path_record:
        raise HTTPException(status_code=404, detail="Learning path not found")
    if path_record["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your learning path")

    path_data = json.loads(path_record.get("corrected_path") or path_record["original_path"])
    sessions = path_data.get("learning_path", [])

    target = None
    for s in sessions:
        if s.get("session_number") == session_number:
            target = s
            break
    if not target:
        raise HTTPException(status_code=404, detail=f"Session {session_number} not found")

    # Ingest session content (lazy — builds vectorstore if not already done)
    sid = ingest_session(
        session=target,
        path_id=str(path_id),
        career_goal=path_record["career_goal"],
    )

    # Update progress
    from datetime import datetime, timezone
    update_session_progress(
        path_id, session_number,
        status="in_progress",
        vectorstore_built=1,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    return {
        "session_id": sid,
        "session_number": session_number,
        "title": target.get("title", ""),
        "status": "in_progress",
        "message": "Session content ingested and ready for study.",
    }


@app.post("/api/session/{path_id}/{session_number}/ask")
async def session_ask(path_id: int, session_number: int,
                      req: SessionAskRequest, user=Depends(get_current_user)):
    """Ask a doubt during a study session — session-scoped RAG query."""
    path_record = get_learning_path(path_id)
    if not path_record:
        raise HTTPException(status_code=404, detail="Learning path not found")

    sid = session_id_for(str(path_id), session_number)
    pipeline = _get_learning_pipeline(path_id)

    try:
        result = pipeline.session_query(sid, req.question)
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_number} has not been started yet. "
                   "Call /api/session/{path_id}/{session_number}/start first."
        )

    return {
        "answer": result.answer,
        "source": result.source,
        "metadata": result.metadata,
    }


@app.post("/api/session/{path_id}/{session_number}/complete")
async def complete_session(path_id: int, session_number: int,
                           user=Depends(get_current_user)):
    """Mark a session as completed."""
    from datetime import datetime, timezone
    update_session_progress(
        path_id, session_number,
        status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    return {"session_number": session_number, "status": "completed"}


@app.get("/api/session/{path_id}/{session_number}/materials")
async def session_materials(path_id: int, session_number: int,
                            user=Depends(get_current_user)):
    """Return the list of sources ingested for a session."""
    sid = session_id_for(str(path_id), session_number)
    materials = get_session_materials(sid)

    if not materials:
        # Session may have been ingested in a prior server run — check vectorstore
        from vector_store import session_vectorstore_exists
        if session_vectorstore_exists(sid):
            return {
                "session_id": sid,
                "ingested": True,
                "content_sources": {},
                "source_details": [],
                "total_chunks": 0,
                "note": "Session was ingested in a previous server session. "
                        "Restart the session to refresh materials metadata.",
            }
        raise HTTPException(status_code=404,
                            detail="Session not ingested yet. Start the session first.")

    return {
        "session_id": sid,
        "ingested": True,
        "content_sources": materials["content_sources"],
        "source_details": materials["source_details"],
        "total_chunks": materials["total_chunks"],
    }


# ── Quiz API ─────────────────────────────────────────────────────────

# In-memory quiz store (per user session — simple approach for now)
_active_quizzes: dict[str, object] = {}


@app.post("/api/quiz/{path_id}/{session_number}/generate")
async def generate_quiz(path_id: int, session_number: int,
                        user=Depends(get_current_user)):
    """Generate a quiz for a session."""
    path_record = get_learning_path(path_id)
    if not path_record:
        raise HTTPException(status_code=404, detail="Learning path not found")

    path_data = json.loads(path_record.get("corrected_path") or path_record["original_path"])
    target = None
    for s in path_data.get("learning_path", []):
        if s.get("session_number") == session_number:
            target = s
            break
    if not target:
        raise HTTPException(status_code=404, detail=f"Session {session_number} not found")

    sid = session_id_for(str(path_id), session_number)
    critic = _get_critic()

    quiz = generate_session_quiz(sid, target, critic)

    # Store for later submission
    quiz_key = f"{user['id']}_{path_id}_{session_number}"
    _active_quizzes[quiz_key] = quiz

    # Build response (without correct answers)
    questions = []
    for q in quiz.questions:
        q_data = {
            "question_text": q.question_text,
            "type": q.question_type,
            "options": q.options,
            "difficulty": q.difficulty,
            "source": q.source,
        }
        if q.reference_text:
            q_data["reference_text"] = q.reference_text
        questions.append(q_data)

    return {
        "session_id": sid,
        "session_number": session_number,
        "total_questions": len(questions),
        "mcq_count": quiz.total_mcq,
        "open_count": quiz.total_open,
        "questions": questions,
    }


@app.post("/api/quiz/{path_id}/{session_number}/submit")
async def submit_quiz(path_id: int, session_number: int,
                      req: QuizSubmitRequest, user=Depends(get_current_user)):
    """Submit quiz answers and get grading results."""
    quiz_key = f"{user['id']}_{path_id}_{session_number}"
    quiz = _active_quizzes.get(quiz_key)
    if not quiz:
        raise HTTPException(
            status_code=400,
            detail="No active quiz found. Generate a quiz first."
        )

    if len(req.answers) != len(quiz.questions):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(quiz.questions)} answers, got {len(req.answers)}"
        )

    critic = _get_critic()
    result = grade_quiz(
        quiz=quiz,
        user_answers=req.answers,
        critic=critic,
        user_id=user["id"],
        path_id=path_id,
    )

    # Clean up active quiz
    del _active_quizzes[quiz_key]

    return {
        "session_number": session_number,
        "attempt_id": result.attempt_id,
        "total_score": result.total_score,
        "max_score": result.max_score,
        "percentage": round(result.percentage, 1),
        "per_skill_scores": result.per_skill_scores,
        "answers": [
            {
                "question": a.question_text,
                "type": a.question_type,
                "source": a.question_source,
                "user_answer": a.user_answer,
                "correct_answer": a.correct_answer,
                "score": a.score,
                "feedback": a.feedback,
            }
            for a in result.answers
        ],
    }


@app.get("/api/quiz/{path_id}/{session_number}/results")
async def quiz_results(path_id: int, session_number: int,
                       user=Depends(get_current_user)):
    """Get past quiz results for a session."""
    attempts = get_quiz_attempts(path_id, session_number, user["id"])
    results = []
    for attempt in attempts:
        answers = get_quiz_answers(attempt["id"])
        results.append({
            "attempt_id": attempt["id"],
            "attempt_number": attempt["attempt_number"],
            "total_score": attempt["total_score"],
            "max_score": attempt["max_score"],
            "percentage": attempt["percentage"],
            "created_at": attempt["created_at"],
            "answers": [dict(a) for a in answers],
        })
    return {"session_number": session_number, "attempts": results}


# ── Mastery & Adaptation API ─────────────────────────────────────────

@app.get("/api/mastery/{path_id}")
async def mastery_scores(path_id: int, user=Depends(get_current_user)):
    """Get mastery summary for a learning path."""
    summary = get_mastery_summary(user["id"], path_id)
    return {
        "path_id": path_id,
        "total_skills": summary.total_skills,
        "mastered": summary.mastered_count,
        "review": summary.review_count,
        "weak": summary.weak_count,
        "not_assessed": summary.not_assessed_count,
        "overall_percentage": round(summary.overall_percentage, 1),
        "skills": [
            {
                "skill": s.skill_label,
                "session_number": s.session_number,
                "score": round(s.mastery_score, 1),
                "attempts": s.attempts_count,
                "status": s.status,
            }
            for s in summary.skills
        ],
    }


@app.post("/api/path/{path_id}/adapt")
async def adapt_learning_path(path_id: int, user=Depends(get_current_user)):
    """Generate an adapted learning path based on mastery scores."""
    path_record = get_learning_path(path_id)
    if not path_record:
        raise HTTPException(status_code=404, detail="Learning path not found")
    if path_record["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your learning path")

    corrected_path = json.loads(
        path_record.get("corrected_path") or path_record["original_path"]
    )
    corrected_path["_filename"] = Path(path_record["filename"]).stem

    adapted_data, adaptation_log = adapt_path(
        user_id=user["id"],
        path_id=path_id,
        corrected_path=corrected_path,
        career_goal=path_record["career_goal"],
    )

    return {
        "path_id": path_id,
        "original_sessions": len(corrected_path.get("learning_path", [])),
        "adapted_sessions": len(adapted_data.get("learning_path", [])),
        "adaptation_log": adaptation_log,
        "adapted_path": adapted_data,
    }
