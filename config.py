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

# ── Ollama / Gemma (local SLM) ──
OLLAMA_MODEL_NAME = "gemma3:4b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"
