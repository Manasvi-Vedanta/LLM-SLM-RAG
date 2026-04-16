"""
config.py
---------
Central configuration for the Self-Correcting RAG System.
All tuneable hyper-parameters and paths live here so every other
module stays free of magic numbers.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env file into os.environ

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"          # persisted FAISS index

# ── Learning Path Pipeline Paths ──
LEARNING_PATH_DIR = BASE_DIR / "Learning Path Inputs"
CORRECTED_PATH_DIR = BASE_DIR / "Corrected Paths"
SESSION_VECTORSTORE_DIR = BASE_DIR / "session_vectorstores"
SESSION_CONTENT_DIR = BASE_DIR / "Session Content"  # user-provided PDFs per session

# ──────────────────────────────────────────────
# PDF Ingestion / Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = 1000         # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
# BAAI/bge-base-en-v1.5: 768-dim retrieval-optimised model.
# Scores ~12 pts higher on MTEB retrieval benchmarks than MiniLM.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# BGE models benefit from a query-instruction prefix (documents don't need one).
EMBEDDING_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ──────────────────────────────────────────────
# Retrieval (Actor) Settings
# ──────────────────────────────────────────────
TOP_K = 5                             # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.20           # cosine-similarity floor (scope gate)
# NOTE: We compute cosine similarity from raw FAISS L2 distances as
# cos_sim = 1 - (L2^2 / 2) for unit-normalised embeddings.
# Relevant chunks typically score 0.20-0.60.  A threshold of 0.20
# keeps useful results while still filtering true misses.
# Tune this after inspecting your own score distribution.

# ──────────────────────────────────────────────
# Critic / Validator Settings
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 85             # percentage above which we trust the excerpt

# ── Backend selection ──
# "gemini"  → Google Gemini 2.5 Flash (cloud LLM) — requires GEMINI_API_KEY
# "gemma"   → Gemma 3 4B (local SLM via Ollama)   — requires Ollama running
CRITIC_BACKEND = "gemma"              # ← change to "gemini" to use the cloud LLM

# ── Gemini API (cloud LLM) ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Fallback models used when the primary hits a rate limit (429).
# The system tries them in order: primary → fallback_1 → fallback_2.
GEMINI_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
]

# ── Ollama / Gemma (local SLM) ──
# Use "gemma3:4b"       for the base (non-fine-tuned) model
# Use "gemma3-critic"   for the fine-tuned model (after running finetune.py)
OLLAMA_MODEL_NAME = "gemma3-critic-v3-new"            # fine-tuned: validation only
OLLAMA_BASE_MODEL = "gemma3:4b"                        # base model: all generation tasks
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ──────────────────────────────────────────────
# Transcript Extraction (optional)
# ──────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"               # "tiny", "base", "small", "medium", "large"

# ──────────────────────────────────────────────
# Web Resource Resolver
# ──────────────────────────────────────────────
WEB_SCRAPE_TIMEOUT = 15                   # seconds per page
MAX_RESOURCES_PER_SESSION = 5             # cap on web searches per session

# ──────────────────────────────────────────────
# Quiz Generation
# ──────────────────────────────────────────────
QUIZ_MCQ_COUNT = 5                        # MCQ questions per quiz
QUIZ_OPEN_COUNT = 3                       # open-ended questions per quiz

# ──────────────────────────────────────────────
# Mastery Thresholds
# ──────────────────────────────────────────────
MASTERY_THRESHOLD_SKIP = 85               # score above which a skill is "mastered"
MASTERY_THRESHOLD_REVIEW = 50             # score below which remediation is needed

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"
