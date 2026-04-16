# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Adaptive Learning Assistant built on a Self-Correcting RAG system with Actor-Critic architecture. Two modes of operation:

1. **RAG Q&A Mode** — Users ask questions about PDF documents; the system retrieves relevant chunks via FAISS, validates with a Critic model, and returns document-grounded or general-knowledge answers.
2. **Learning Path Pipeline** — Users load a career-oriented learning path JSON, the system validates/corrects it, ingests content per-session, enables study with doubt clarification, generates document-grounded quizzes, tracks mastery, and adapts the path based on performance.

## Commands

```bash
# Run web UI (all features)
uvicorn server:app --host 127.0.0.1 --port 8000

# Original RAG Q&A mode (CLI)
python main.py                # interactive loop with configured critic
python main.py --rebuild      # force FAISS index rebuild
python main.py --mock         # offline testing with MockCritic

# Learning path pipeline (CLI)
python main.py --load-path "ML Engineer.json"   # load + validate path
python main.py --sessions                       # list sessions with status
python main.py --study 3                        # interactive study mode
python main.py --quiz 3                         # take quiz for session 3
python main.py --mastery                        # view mastery summary
python main.py --adapt                          # generate adapted path

# Evaluate critic backends
python evaluate.py

# Fine-tune (requires NVIDIA GPU, ~6GB VRAM)
pip install -r requirements-finetune.txt
python finetune.py --data training_data_v3.jsonl --output-dir finetuned_model_v3
```

## Architecture

### Two LLM Roles
- **Fine-tuned model** (`gemma3-critic-v3-new` via Ollama): Validation and confidence scoring ONLY. Configured as `OLLAMA_MODEL_NAME`.
- **Base model** (`gemma3:4b` via Ollama): All generation tasks — path correction, question generation, answer evaluation, content generation, resource suggestions. Configured as `OLLAMA_BASE_MODEL`.
- **Cloud fallback** (Gemini): Used only when local model fails. No GPT or other paid APIs.

### Core RAG Pipeline
- `config.py` — All tuneable parameters. No magic numbers elsewhere.
- `llm_service.py` — Shared LLM call utilities (`call_ollama`, `call_gemini`, `call_llm` dispatcher). Used by critic and all pipeline components.
- `ingestion.py` — PDF, text content, and transcript loading + chunking.
- `vector_store.py` — FAISS index with BGE embeddings. Global index + per-session indexes.
- `critic.py` — Abstract `BaseCritic` with `GeminiCritic`, `GemmaCritic`, `MockCritic`. Methods: `validate`, `format_answer`, `generate_fallback_answer`, `generate_questions`, `evaluate_answer`.
- `pipeline.py` — `RAGPipeline` with `query()` (global) and `session_query()` (session-scoped).

### Learning Path Pipeline (6 steps, strict order)

**Step 1 — Path Validation:**
- `path_validator.py` — Loads JSON from `Learning Path Inputs/`, extracts career goal from filename, LLM-validates each session, auto-corrects mismatches, saves to `Corrected Paths/`.

**Step 2 — Content Ingestion (lazy, per-session):**
- `web_resource_resolver.py` — Resolves resource names to URLs via DuckDuckGo, scrapes readable content.
- `transcript_extractor.py` — Optional yt-dlp + Whisper pipeline. Skips gracefully if not installed. `is_video_url()` detects video URLs across YouTube, Vimeo, Dailymotion, etc.
- `session_mapper.py` — Collects content per session in priority: user PDFs > scraped web > comprehensive_guides > video transcripts. Tracks `source_details` (individual source metadata) and `content_sources` (counts). Also checks resolved URLs from web scraping for video content.
- `session_orchestrator.py` — Single entry point: `ingest_session()` → maps content → builds per-session FAISS index. Caches materials metadata via `get_session_materials()` for the materials API endpoint.
- Per-session FAISS indexes stored in `session_vectorstores/{path_id}_session_{n}/`.

**Step 3 — Study & Doubt Clarification:**
- `pipeline.py: session_query()` — Queries session-specific vector store, same Actor->Critic flow.

**Step 4 — Quiz Generation & Evaluation:**
- `question_generator.py` — Generates document-grounded MCQ + open-ended questions. Critic validates each candidate. Supplements with general-knowledge questions (labeled) if content is thin. `QuizQuestion` has `reference_text` field for code/excerpt that the student needs to see alongside the question.
- `answer_evaluator.py` — MCQ: direct grading. Open-ended: Critic scores against source excerpts. Saves to DB.
- Quiz generation prompt instructs LLM to never use "from the excerpt below" phrasing; instead, relevant material goes into `reference_text`.

**Step 5 — Mastery Tracking:**
- `mastery_tracker.py` — Weighted running average (70% new, 30% old). Classifies: mastered (>85%), review (50-85%), weak (<50%).

**Step 6 — Path Adaptation:**
- `path_adapter.py` — Mastered sessions compressed, weak sessions get remediation injected, resources auto-suggested. Outputs adapted JSON.

### Web Backend
- `server.py` — FastAPI. 30 API routes: auth (3), original chat (2), path management (5), session (4 — includes materials endpoint), quiz (3), mastery/adaptation (2), health (1), pages (5 — `/`, `/login`, `/signup`, `/chat`, `/learn`).
- `auth.py` — JWT + bcrypt.
- `database.py` — SQLite WAL. 7 tables: `users`, `chat_history`, `learning_paths`, `session_progress`, `quiz_attempts`, `quiz_answers`, `skill_mastery`.

### Frontend
- `static/learn.html` — Learning pipeline UI: sidebar (path selector, session list with status icons), study chat panel, quiz panel (renders `reference_text` as code blocks or blockquotes), mastery dashboard with per-skill progress bars, adaptation log panel, materials modal showing all ingested sources grouped by type.
- `static/chat.html` — RAG Q&A chatbot interface.
- `static/index.html` — Landing page with "Learn" and "Chat" nav links (visible when logged in).
- `static/css/learn.css` — Learning page styles including modal, quiz reference blocks, mastery bars.
- `static/css/style.css` — Global dark theme.
- `static/js/auth.js` — Token management, authenticated fetch wrapper, route guards.
- `static/js/background.js` — Three.js animated gradient mesh.
- No build step. Vanilla HTML/CSS/JS.

### Fine-Tuning Pipeline
- `finetune.py`, `generate_training_data.py`, `generate_synthetic_data.py`, `merge_lora_cpu.py`, `evaluate.py`

## Key Design Decisions

- **Dual thresholds:** Similarity gate (0.20) + confidence gate (85%). Configurable per-pipeline instance.
- **Two-model strategy:** Fine-tuned model for validation only; base model for all generation. Prevents degraded output from using the wrong model.
- **Per-session vector stores:** Each session has its own FAISS index, built lazily on first access. Scopes retrieval to relevant content only.
- **Content ingestion priority:** User PDFs > scraped web content > JSON guides > video transcripts. Graceful degradation at each level.
- **Quiz source labeling:** Every question tagged `"your_materials"` or `"general_knowledge"`. User always knows what they're being tested on.
- **Quiz reference material:** Questions that depend on code or passages include the material in `reference_text`, displayed alongside the question. The generation prompt forbids "from the excerpt below" phrasing.
- **Materials provenance:** `session_mapper.py` tracks individual source details (type, name, URL, path) in `source_details`, cached by `session_orchestrator.py` and served via `GET /api/session/{path_id}/{n}/materials`.
- **Path never overwritten:** Original JSON untouched. Corrected and adapted versions saved as separate files.
- **Video URL detection:** `transcript_extractor.is_video_url()` checks both raw resource names and resolved URLs from web scraping, ensuring YouTube links discovered during DuckDuckGo resolution also get transcript extraction.

## Environment

- Python 3.10+. Dependencies: `requirements.txt` (core), `requirements-finetune.txt` (GPU).
- `.env` file for API keys. Gitignored.
- Learning path JSONs in `Learning Path Inputs/` (20 files, e.g. `ML Engineer.json`).
- PDFs for original RAG mode in `Dataset/`.
- User session PDFs in `Session Content/{path_id}/session_{n}/`.
- Per-session FAISS indexes in `session_vectorstores/`.
- Corrected/adapted paths in `Corrected Paths/`.
- `users.db` — SQLite database (gitignored).
- Ollama must be running with both `gemma3:4b` (base) and `gemma3-critic-v3-new` (fine-tuned).
