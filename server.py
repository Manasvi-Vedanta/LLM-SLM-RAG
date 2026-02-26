"""
server.py – FastAPI backend for the Self-Correcting RAG Web UI
===============================================================

Endpoints
~~~~~~~~~
POST /api/auth/signup      – register a new user
POST /api/auth/login       – authenticate and receive JWT
GET  /api/auth/me          – get current user info
POST /api/chat             – send a question to the RAG pipeline
GET  /api/chat/history     – retrieve chat history
GET  /                     – landing page
GET  /login                – login page
GET  /signup               – signup page
GET  /chat                 – chatbot page

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
)
from ingestion import ingest
from vector_store import build_vectorstore, load_vectorstore
from critic import GeminiCritic, GemmaCritic, MockCritic, create_critic
from pipeline import RAGPipeline

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


@app.on_event("startup")
async def startup_event():
    global _pipeline
    logger.info("Loading RAG pipeline...")

    # Build or load vector store
    index_path = config.VECTORSTORE_DIR / "index.faiss"
    if index_path.exists():
        logger.info("Loading existing FAISS index...")
        vectorstore = load_vectorstore()
    else:
        logger.info("Building FAISS index from PDFs...")
        chunks = ingest()
        logger.info("Ingested %d chunks", len(chunks))
        vectorstore = build_vectorstore(chunks)

    # Initialise critic (reads CRITIC_BACKEND from config.py)
    critic = create_critic()
    logger.info(
        "Critic ready: %s (backend=%s)",
        type(critic).__name__,
        config.CRITIC_BACKEND,
    )

    _pipeline = RAGPipeline(vectorstore=vectorstore, critic=critic)
    logger.info("RAG pipeline ready.")


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
