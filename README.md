# Self-Correcting RAG System with Actor-Critic Architecture

A full-stack Retrieval-Augmented Generation system with two modes: (1) **RAG Q&A** — ask questions about PDF documents with Actor-Critic validated answers, and (2) **Adaptive Learning Pipeline** — load career-oriented learning paths, study with doubt clarification, take document-grounded quizzes, track skill mastery, and get AI-adapted learning paths. Uses a swappable **Critic** model (cloud LLM **or** fine-tuned local SLM) and serves everything through a professional web interface with user authentication.

---

## Key Features

### RAG Q&A Mode
- **Actor-Critic RAG pipeline** — retrieves document chunks, then validates them with a Critic model before responding
- **Three answer modes** — formatted document answer, general-knowledge fallback, or out-of-scope rejection
- **Structured answer formatting** — high-confidence document excerpts are restructured into clean Definition / Syntax / Key Points / Example format via the base Gemma model

### Adaptive Learning Pipeline
- **Learning path validation** — loads career-oriented JSON learning paths, LLM-validates each session's skills and resources, auto-corrects mismatches
- **Per-session content ingestion** — resolves resource names to URLs via multi-backend web search (DDGS, Brave, DuckDuckGo HTML), scrapes content with trafilatura, auto-detects video URLs and routes them to transcript extraction, builds per-session FAISS indexes
- **Study mode with doubt clarification** — session-scoped RAG Q&A against ingested materials
- **Document-grounded quizzes** — generates MCQ + open-ended questions from session materials, grades answers against source excerpts
- **Skill mastery tracking (two-stage)** — on quiz submit, the stored `mastery_score` is updated as a weighted running average (70% new / 30% old) against the prior raw score. On every read, the stored score is passed through an **Ebbinghaus exponential time-decay** to produce a `decayed_score`; all downstream consumers (path adapter, prerequisite readiness check, review scheduler) use the decayed score so stale mastery is surfaced automatically. Classification on the decayed score: mastered (>85%) / review (50–85%) / weak (<50%), plus a `needs_review` flag when `raw ≥ 85 AND decayed < 85`.
- **Adaptive path generation** — uses decayed mastery to compress mastered sessions, inject remediation for weak skills, and suggest additional resources; then runs a graph-constrained topological reorder to heal any prerequisite violations introduced by remediation injection

### Adaptive Learning Analytics (Research-Grade)
- **ESCO skill prerequisite graph** — builds a directed acyclic graph (DAG) from session prerequisites, supports topological ordering, transitive prerequisite chains, dependent-skill lookup, and graph-constrained session reordering; path adapter uses it to heal prerequisite violations after remediation injection
- **Ebbinghaus mastery decay** — per-skill mastery scores exponentially decay with time-since-last-assessment; the adapter uses *decayed* scores (not raw) so a stale 95% rightly triggers review
- **Spaced repetition scheduler** — derives per-skill review intervals from the Ebbinghaus forgetting curve (`d = ln(S/T)/λ`), classifies items as overdue / due today / due this week / scheduled
- **Knowledge-transfer metrics** — pairs prerequisite mastery with dependent-session performance, computes Pearson correlation, flags broken chains (high prereq mastery → weak downstream) and strong chains
- **Bloom's Taxonomy + ZPD quiz targeting** — every generated question is labelled with its cognitive level (remember → create); the target level window is picked from current mastery so the quiz sits one rung above the learner's comfort zone (Vygotsky's Zone of Proximal Development)
- **Metacognitive calibration** — students self-rate confidence before a quiz; the system compares predicted vs. actual to surface over-/under-confidence patterns (Dunning-Kruger signal) with Pearson correlation
- **Cognitive-load signals** — tracks doubt-query count and study duration per session as proxies for intrinsic + extraneous load (Sweller); flags high-friction sessions
- **Chunk-level citation provenance** — every document-grounded answer carries a citation array (source file, page, similarity score, 240-char preview) of every passing chunk — one click from answer to exact supporting excerpt

### Infrastructure
- **LLM / SLM selection** — switch between Google Gemini 2.5 Flash (cloud LLM) and a fine-tuned Gemma 3 4B (local SLM via Ollama) with a single config change
- **Two-model strategy** — fine-tuned model for validation/confidence scoring only; base model for all generation tasks (path correction, question generation, answer evaluation, content generation)
- **Complete fine-tuning pipeline** — synthetic data generation, QLoRA training, GGUF conversion, Ollama deployment, and side-by-side evaluation
- **Rate-limit resilient** — Gemini backend cascades through fallback models (`gemini-2.5-flash` → `gemini-3-flash-preview` → `gemini-2.5-flash-lite`) on 429 errors
- **Web UI** — dark-themed glassmorphism design with Three.js animated gradient-mesh background
- **Authentication** — JWT-based signup/login with bcrypt password hashing and per-user chat history
- **Two interfaces** — browser-based web UI (FastAPI) and terminal CLI
- **Fully configurable** — all thresholds, models, backends, and chunk parameters in one config file

---

## Architecture

```
  User Question
       │
       ▼
  ┌─────────────┐     top-k chunks     ┌──────────────┐
  │  FAISS Actor │ ──────────────────►  │  Scope Gate  │
  │  (Retriever) │                      │  sim < 0.20? │
  └─────────────┘                      └──────┬───────┘
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                        Out of Scope               ┌──────────────────┐
                                                   │  Critic Model    │
                                                   │  (Gemini / Gemma)│
                                                   └────────┬─────────┘
                                                            │
                                             ┌──────────────┴──────────────┐
                                             ▼                             ▼
                                    confidence > 85%              confidence ≤ 85%
                                    Format excerpt into           LLM/SLM fallback
                                    structured answer             from general knowledge
```

### Module Breakdown

#### Core RAG Pipeline

| Module | Role |
|---|---|
| `config.py` | All tuneable parameters — paths, models, thresholds, backend selection |
| `llm_service.py` | Shared LLM call utilities — `call_ollama`, `call_gemini`, `call_llm` dispatcher; routes validation to fine-tuned model, generation to base model |
| `ingestion.py` | PDF, text content, and transcript loading + chunking |
| `vector_store.py` | FAISS index with BGE embeddings — global index + per-session indexes |
| `critic.py` | `GeminiCritic`, `GemmaCritic`, `MockCritic` — validation, formatting, question generation, answer evaluation |
| `pipeline.py` | RAG pipeline with `query()` (global) and `session_query()` (session-scoped) |

#### Learning Path Pipeline

| Module | Role |
|---|---|
| `path_validator.py` | Loads JSON from `Learning Path Inputs/`, LLM-validates each session, auto-corrects mismatches, saves to `Corrected Paths/` |
| `web_resource_resolver.py` | Multi-backend URL resolution (DDGS → Brave → DDG HTML), content extraction (trafilatura → BS4), auto-detects and separates video URLs from web content |
| `transcript_extractor.py` | Optional yt-dlp + Whisper pipeline for video transcripts; `is_video_url()` detects YouTube, Vimeo, Dailymotion, etc. (skips gracefully if not installed) |
| `session_mapper.py` | Collects content per session in priority: user PDFs > scraped web > comprehensive guides > video transcripts. Routes video URLs to transcript extraction; shows unavailable transcripts as direct links in materials |
| `session_orchestrator.py` | Single entry point: `ingest_session()` → maps content → builds per-session FAISS index. Caches materials metadata (sources, provenance) for the materials API |
| `question_generator.py` | Generates document-grounded MCQ + open-ended questions; Critic validates each candidate; supplements with general-knowledge questions (labelled) if content is thin |
| `answer_evaluator.py` | MCQ: direct grading. Open-ended: Critic scores against source excerpts. Saves results to database |
| `mastery_tracker.py` | **Write-side:** weighted running average (70% new, 30% old) on quiz submit. **Read-side:** Ebbinghaus exponential time-decay `decayed = max(FLOOR, raw·e^(-λ·days))` applied on every read. Classifies mastered (>85%) / review (50–85%) / weak (<50%) against the decayed score; flags `needs_review` when raw ≥ 85 but decayed drops below |
| `path_adapter.py` | Mastered sessions compressed, weak sessions get remediation injected, resources auto-suggested. Outputs adapted JSON |
| `skill_graph.py` | Directed skill prerequisite DAG — topological sort, prerequisite chains, ESCO URI mapping, graph-constrained reorder |
| `knowledge_transfer.py` | Pearson correlation between prerequisite mastery and dependent-session performance; flags weak/strong transfer chains |
| `review_scheduler.py` | Spaced-repetition review queue derived from the Ebbinghaus forgetting curve; classifies urgency (overdue / due today / this week / scheduled) |

#### Fine-Tuning Pipeline

| Module | Role |
|---|---|
| `generate_training_data.py` | Gemini-based teacher labelling — runs queries through Gemini to produce labelled JSONL |
| `generate_synthetic_data.py` | Heuristic synthetic data generator — creates training data without API calls using similarity/keyword heuristics |
| `finetune.py` | QLoRA fine-tuning of Gemma 3 4B using PEFT + BitsAndBytes + TRL |
| `merge_lora_cpu.py` | Merges LoRA adapter into base model on CPU — produces clean FP16 weights for GGUF conversion |
| `evaluate.py` | Side-by-side benchmark comparing LLM vs SLM on accuracy, latency, and agreement |

#### Backend & Auth

| Module | Role |
|---|---|
| `server.py` | FastAPI backend — 34 API routes: auth (3), original chat (2), path management (5), session (4 — includes materials endpoint), quiz (3), mastery/adaptation (2), review-schedule (1), transfer (1), calibration (1), cognitive-load (1), health (1), pages (5) |
| `auth.py` | JWT token creation/verification, bcrypt password hashing |
| `database.py` | SQLite WAL mode — 7 tables: `users`, `chat_history`, `learning_paths`, `session_progress` (incl. `doubt_query_count` + `study_duration_seconds` for cognitive-load signals), `quiz_attempts` (incl. `self_confidence` for calibration), `quiz_answers`, `skill_mastery` (incl. `last_assessed_at` for review scheduling) |
| `main.py` | Terminal CLI — original RAG Q&A loop + 6 learning pipeline commands |

### Frontend

| File | Purpose |
|---|---|
| `static/index.html` | Landing page with feature cards and hero section |
| `static/login.html` | Login form with validation and error display |
| `static/signup.html` | Registration form with password confirmation |
| `static/chat.html` | Chatbot interface with typing indicators, source badges, suggestion chips, chat history |
| `static/learn.html` | Learning pipeline UI — path loading, session sidebar, study chat, materials modal (grouped by source type with provenance), quiz with reference text blocks, mastery dashboard, path adaptation |
| `static/js/background.js` | Three.js animated gradient mesh — undulating surface with layered sine waves, floating orbs, mouse-reactive camera |
| `static/js/auth.js` | Token management, authenticated fetch wrapper, route guards |
| `static/css/style.css` | Dark theme with CSS variables, glassmorphism cards, responsive layout, animations |
| `static/css/learn.css` | Learning pipeline page styles — sidebar, study chat, quiz cards, mastery bars, responsive layout |

---

## Project Structure

```text
Code/
├── Dataset/                      # PDFs for original RAG Q&A mode
├── Learning Path Inputs/         # Career-oriented learning path JSONs (20 files)
├── Corrected Paths/              # Validated/corrected/adapted path outputs
├── Session Content/              # User PDFs per session (path_id/session_n/)
├── session_vectorstores/         # Per-session FAISS indexes (generated)
├── static/
│   ├── css/
│   │   ├── style.css             # Global dark theme
│   │   └── learn.css             # Learning pipeline page styles
│   ├── js/
│   │   ├── auth.js
│   │   └── background.js
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── chat.html
│   └── learn.html                # Learning pipeline UI
├── vectorstore/                  # Global FAISS index (generated, gitignored)
├── finetuned_model*/             # Fine-tune artifacts (gitignored)
│
│── # ── Core RAG ──
├── config.py                     # All tuneable parameters
├── llm_service.py                # Shared LLM call dispatcher
├── ingestion.py                  # PDF/text/transcript loading + chunking
├── vector_store.py               # FAISS index (global + per-session)
├── critic.py                     # Critic implementations (Gemini/Gemma/Mock)
├── pipeline.py                   # RAG pipeline (global + session-scoped)
│
│── # ── Learning Pipeline ──
├── path_validator.py             # Path validation & auto-correction
├── web_resource_resolver.py      # URL resolution + web scraping
├── transcript_extractor.py       # YouTube transcript extraction (optional)
├── session_mapper.py             # Session-document content mapping
├── session_orchestrator.py       # Per-session ingestion orchestrator
├── question_generator.py         # Quiz question generation
├── answer_evaluator.py           # Answer evaluation & grading
├── mastery_tracker.py            # Skill mastery tracking
├── path_adapter.py               # Adaptive path generation
├── skill_graph.py                # Prerequisite DAG + ESCO mapping
├── knowledge_transfer.py         # Cross-session transfer correlation
├── review_scheduler.py           # Ebbinghaus-based review scheduler
│
│── # ── Fine-Tuning ──
├── generate_training_data.py
├── generate_synthetic_data.py
├── finetune.py
├── merge_lora_cpu.py
├── evaluate.py
├── Modelfile
│
│── # ── Backend & Auth ──
├── server.py                     # FastAPI (34 routes)
├── auth.py                       # JWT + bcrypt
├── database.py                   # SQLite (7 tables)
├── main.py                       # CLI entry point
│
├── requirements.txt
├── requirements-finetune.txt
├── .env                          # API keys (gitignored)
└── README.md
```

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- **For Gemini backend:** Internet access + Gemini API key
- **For Gemma backend:** [Ollama](https://ollama.com) installed locally
- **For fine-tuning:** NVIDIA GPU with ≥ 6 GB VRAM + CUDA

---

## Setup

### 1) Clone

```bash
git clone https://github.com/Manasvi-Vedanta/LLM-SLM-RAG.git
cd LLM-SLM-RAG
```

### 2) Virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure the Critic backend

Open `config.py` and set `CRITIC_BACKEND`:

```python
# "gemini"  → Google Gemini 2.5 Flash (cloud LLM)
# "gemma"   → Gemma 3 4B (local SLM via Ollama)
CRITIC_BACKEND = "gemma"    # or "gemini"
```

#### Option A — Gemini (cloud LLM)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Option B — Gemma 3 4B (local SLM)

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the base model: `ollama pull gemma3:4b`
3. Deploy the fine-tuned critic model (see [Fine-Tuning](#fine-tuning-the-slm--theory--practice) or download from [HuggingFace](#pre-trained-models-huggingface))
4. Ensure Ollama is running (`ollama serve` or the system tray app)
5. Set `CRITIC_BACKEND = "gemma"` in `config.py`

**Both models are required:** `gemma3:4b` (base) handles all generation tasks (path correction, question generation, answer evaluation, content generation), while `gemma3-critic-v3-new` (fine-tuned) handles validation and confidence scoring only.

No API key or internet is needed for inference once the models are downloaded.

---

## Run

### Web UI (recommended)

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

| Route | Page |
|---|---|
| `/` | Landing page |
| `/signup` | Create an account |
| `/login` | Sign in |
| `/chat` | Chatbot — RAG Q&A (requires login) |
| `/learn` | Learning pipeline — paths, study, quizzes, mastery (requires login) |

### CLI mode

```bash
# Original RAG Q&A
python main.py               # interactive terminal loop
python main.py --rebuild     # force-rebuild the FAISS index
python main.py --mock        # use MockCritic (no API key needed)
python main.py --threshold 0.25 --confidence 90   # override gates

# Learning path pipeline
python main.py --load-path "ML Engineer.json"   # load + validate a path
python main.py --sessions                       # list sessions with status
python main.py --study 3                        # interactive study mode for session 3
python main.py --quiz 3                         # take quiz for session 3
python main.py --mastery                        # view mastery summary
python main.py --adapt                          # generate adapted path
```

---

## Configuration

All parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CRITIC_BACKEND` | `"gemma"` | `"gemini"` for cloud LLM, `"gemma"` for local SLM, `"mock"` for offline |
| `CHUNK_SIZE` | 1000 | Characters per document chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model (768 dims) |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.20 | Cosine similarity floor for scope gate |
| `CONFIDENCE_THRESHOLD` | 85 | Critic confidence floor (0–100) |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash` | Primary cloud LLM model |
| `GEMINI_FALLBACK_MODELS` | `[gemini-3-flash-preview, gemini-2.5-flash-lite]` | Fallback models on rate limit |
| `OLLAMA_MODEL_NAME` | `gemma3-critic-v3-new` | Fine-tuned SLM (validation only) |
| `OLLAMA_BASE_MODEL` | `gemma3:4b` | Base model (all generation tasks) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `QUIZ_MCQ_COUNT` | 5 | MCQ questions per quiz |
| `QUIZ_OPEN_COUNT` | 3 | Open-ended questions per quiz |
| `MASTERY_THRESHOLD_SKIP` | 85 | Score above which a skill is "mastered" |
| `MASTERY_THRESHOLD_REVIEW` | 50 | Score below which remediation is needed |
| `WEB_SCRAPE_TIMEOUT` | 15 | Seconds per page for web scraping |
| `MAX_RESOURCES_PER_SESSION` | 5 | Cap on web searches per session |

**Tuning tips:**
- Lower `SIMILARITY_THRESHOLD` if relevant questions are marked out-of-scope
- Increase `CONFIDENCE_THRESHOLD` for stricter document-only answers
- Increase `TOP_K` if answers seem incomplete
- Adjust `MASTERY_THRESHOLD_SKIP` / `MASTERY_THRESHOLD_REVIEW` to control when path adaptation compresses or injects remediation sessions

---

## How Answers Are Decided

1. **Actor** retrieves top-k document chunks for the query
2. **Scope Gate** — if best cosine similarity < threshold → **Out of Scope**
3. **Critic** — the selected model (Gemini or Gemma) validates whether the excerpt answers the question (confidence 0–100)
4. **Confidence Gate**:
   - ≥ threshold → return **exact document excerpt** with source file, page, and scores
   - < threshold → return **general-knowledge fallback** generated by the Critic

The cosine similarity is computed from FAISS L2 distances on unit-normalised embeddings:

$$\text{cosine\_sim} = 1 - \frac{L2^2}{2}$$

---

## How the Learning Pipeline Works

The learning pipeline follows a strict 6-step progression:

```
  ┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │ 1. Load Path  │ →  │ 2. Validate &    │ →  │ 3. Ingest Content│
  │    (JSON)     │    │    Auto-Correct   │    │    (per-session) │
  └───────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                       ┌────────────────────────────────┘
                       ▼
  ┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │ 4. Study +    │ →  │ 5. Quiz &        │ →  │ 6. Mastery →     │
  │    Doubt Q&A  │    │    Grading       │    │    Adapt Path    │
  └───────────────┘    └──────────────────┘    └──────────────────┘
```

1. **Path Loading** — A learning path JSON (e.g., `ML Engineer.json`) defines sessions with skills, objectives, resources, and prerequisites
2. **Validation** — Each session is LLM-validated: skills checked against title/objectives, difficulty verified, mismatches auto-corrected. Corrected path saved separately (original is never modified)
3. **Content Ingestion** (lazy, per-session) — Resolves resource names to URLs via multi-backend search (DDGS → Brave → DDG HTML), scrapes web content with trafilatura, auto-detects video URLs and routes them to transcript extraction (or flags them as direct links if yt-dlp/whisper unavailable). Priority: user PDFs > scraped web > JSON guides > video transcripts. Builds a per-session FAISS index in `session_vectorstores/`
4. **Study Mode** — Session-scoped RAG Q&A using the per-session vector store. Same Actor→Critic flow as the original pipeline
5. **Quiz Generation & Grading** — Generates document-grounded MCQ + open-ended questions (each validated by the Critic). MCQ graded directly; open-ended scored against source excerpts. Every question tagged with its source (`your_materials` or `general_knowledge`)
6. **Mastery & Adaptation** — Per-skill raw mastery is updated via a weighted running average (70% new, 30% old) on every quiz submit, then passed through an Ebbinghaus exponential time-decay on read. Path adaptation consumes the **decayed** score: mastered sessions (>85%) compressed, weak sessions (<50%) get LLM-generated remediation injected, review sessions (50–85%) get additional resources. A graph-constrained reorder heals any prerequisite violations introduced by injection

---

## Critic Backend Details

### Gemini (Cloud LLM)

- **Model:** Google Gemini 2.5 Flash (with automatic fallback to `gemini-3-flash-preview` → `gemini-2.5-flash-lite` on rate limits)
- **Pros:** High accuracy, nuanced scope discrimination, strong instruction-following
- **Cons:** Requires API key, internet, subject to rate limits (automatic retry with exponential backoff built in)

### Gemma 3 4B (Local SLM — Fine-Tuned)

- **Model:** Gemma 3 4B fine-tuned via QLoRA (v3-new), quantised to Q4_K_M, deployed via Ollama as `gemma3-critic-v3-new`
- **Pros:** Fully offline, no API costs, ~5s consistent latency, 100% scope accuracy and 100% document answer rate on in-scope questions
- **Cons:** Requires ~2.5 GB disk, ~4 GB RAM while running

Both backends share identical validation and fallback prompts. The `create_critic()` factory in `critic.py` reads `CRITIC_BACKEND` from config and returns the appropriate implementation.

---

## Evaluation Protocol

A full publication-grade evaluation plan — research questions, hypotheses,
metrics, statistics, ablation matrix, datasets, and threats-to-validity —
is documented in [`EVALUATION.md`](EVALUATION.md). It organises experiments
in four tiers:

| Tier | Unit | Examples | Runnable |
|------|------|----------|----------|
| 1 | Fine-tuned critic SLM | F1 + bootstrap CI, ECE/Brier, κ + McNemar, Qwen head-to-head | `evaluation/tier1_critic_benchmark.py` |
| 2 | End-to-end RAG pipeline | per-sentence faithfulness (self-audit / cross-audit) | `evaluation/rag_self_faithfulness.py` |
| 3 | Offline analytics mechanics | Ebbinghaus monotonicity, scheduler closed-form, ZPD invariants, Pearson recovery | `evaluation/synthetic_user_sim.py` |
| 4 | End-to-end human outcomes | A/B on retention, ZPD-lift, calibration feedback loop | IRB user study |

### Latest results (2026-04-22)

Two 4 B critic fine-tunes trained on the identical `training_data_v3.jsonl` (300 labelled examples, same LoRA config r=16 α=32) were evaluated head-to-head:

| Metric | `gemma3-critic-v3-new` | `qwen-critic-v1` |
|--------|------------------------|------------------|
| **Tier 1** Accuracy (N=25) | **0.84** (CI 0.68–0.96) | 0.64 (CI 0.44–0.80) |
| **Tier 1** F1 | **0.909** (CI 0.80–0.98) | 0.710 (CI 0.50–0.86) |
| **Tier 1** ECE (10-bin) | **0.056** | 0.177 |
| **Tier 1** Brier | 0.135 | **0.041** |
| **Tier 1** Latency p50 / p95 | 7.4 s / 15.1 s | **3.5 s / 3.6 s** |
| **Tier 2** Document answers (N=10) | 10 / 10 | 0 / 10 |
| **Tier 2** Mean faithfulness | **1.00** | — (all fell through gate) |
| **Tier 3** Analytics invariants | 4 / 4 PASS (shared, deterministic) | — |

**Agreement:** Cohen's κ = 0.063, raw agreement 48 %, McNemar χ² = 1.23 (ns).

**Interpretation.** Identical training data on a different 4 B architecture produces dramatically different confidence distributions. Gemma v3 outputs 87–92 on in-scope items (clears the 85-gate), while Qwen clusters at 72–85 (fails the gate) — hence every Tier-2 answer falls through to the general-knowledge fallback. This is a **load-bearing finding for the paper**: Tier 1 ECE (0.056 vs 0.177) directly predicts Tier 2 pipeline behaviour, confirming that critic calibration is the primary mechanism gating retrieval-grounded answers.

Reports written to `evaluation/results/tier{1,2,3}_*.json` (gitignored). Gemini cross-audit remains pre-specified but unrun (cloud rate limits).

Tier 3 is deterministic, LLM-free, and CI-safe — run it before every
commit that touches `mastery_tracker.py`, `review_scheduler.py`,
`question_generator.py`, or `knowledge_transfer.py`:

```bash
python evaluation/synthetic_user_sim.py -v
```

The **Qwen critic head-to-head** slot (E4b in `EVALUATION.md`) is
pre-specified so that a future LoRA fine-tune of Qwen on
`training_data_v3.jsonl` slots into the same comparison table without
re-designing the experiment.

---

## Adaptive Learning Analytics — Theory

Beyond the core mastery tracker and path adapter, the project ships seven research-grade analytics modules grounded in classical cognitive-science and learning-analytics literature. Each is a self-contained module that can be called from CLI, notebooks, or HTTP endpoints.

| # | Module | Grounded in |
|---|---|---|
| 1 | `skill_graph.py` | Graph theory — Kahn's topological sort, DAG validation; ESCO taxonomy |
| 2 | `mastery_tracker.py` (decay) | Ebbinghaus forgetting curve (1885) |
| 3 | `review_scheduler.py` | Spaced-repetition interval derivation (SM-2 family) |
| 4 | `knowledge_transfer.py` | Pearson correlation; transfer-of-learning theory (Thorndike, Perkins & Salomon) |
| 5 | `question_generator.py` (ZPD) | Bloom's Taxonomy (1956, revised 2001); Vygotsky's Zone of Proximal Development (1978) |
| 6 | `database.get_calibration_data` + `/api/calibration` | Metacognition (Flavell, 1979); Dunning-Kruger effect (1999) |
| 7 | `database.get_cognitive_load_data` + `/api/cognitive-load` | Cognitive Load Theory (Sweller, 1988) |
| 8 | `pipeline._build_citations` | RAG faithfulness / provenance (Lewis et al. 2020, Gao et al. 2023) |

### 1. ESCO Skill Prerequisite Graph (`skill_graph.py`)

Learning paths are not flat lists — they form a **directed acyclic graph (DAG)** where each edge `A → B` means "skill A is a prerequisite of skill B". The graph is built from the `prerequisites` field on each session plus the `skill_details` array (which carries ESCO URIs for alignment with the European Skills/Competences/Occupations taxonomy).

**Topological sort.** A valid teaching order is any permutation that respects all prerequisite edges. Kahn's algorithm produces this in O(V+E):

```
repeat:
    pick a node with in-degree 0
    emit it, remove its outgoing edges
until graph is empty (or a cycle is detected)
```

**Use cases exposed by the module:**
- `topological_order()` — canonical learning sequence
- `get_prerequisite_chain(skill)` — transitive prereqs in learning order (DFS on reverse edges)
- `get_dependents(skill)` — all skills that transitively depend on this one
- `validate_ordering(sessions)` — returns violations where a session's prereq was never taught earlier
- `constrained_reorder(sessions, priority_map)` — reorders sessions to respect prereqs while honouring an optional priority (e.g., mastery-weighted)
- `get_esco_mapping()` — flat `skill_label → ESCO URI` dictionary for standards alignment

This formalises what the `path_adapter` does heuristically, and serves as the substrate for future graph-aware adaptation strategies.

### 2. Knowledge-Transfer Metrics (`knowledge_transfer.py`)

A core question in curriculum design: **does mastering a prerequisite actually translate into success on the downstream session?** If prereq mastery is high but dependent performance is low, the dependency graph is *broken* — either the prereq test is too lenient, the dependent material requires something additional, or transfer simply isn't happening.

**Pearson correlation.** For each session with at least one quiz attempt, we pair `(avg_prereq_mastery, session_score)` and compute:

$$r = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sqrt{\sum_i (x_i - \bar x)^2} \sqrt{\sum_i (y_i - \bar y)^2}}$$

`r ≈ 1` means the graph's prereq structure predicts outcomes well; `r ≈ 0` means mastery of prereqs says nothing about downstream performance. The correlation is only reported when n ≥ 3 pairs.

**Per-prerequisite transfer strength.** For each prerequisite skill X:

$$\text{transfer}(X) = \min(\text{mastery}(X), \text{avg dependent score})$$

The `min` captures the idea that *a skill has only transferred as far as the weakest link in its downstream chain*. A skill with 95% mastery whose dependents average only 40% has a transfer score of 40% — a red flag.

**Automatic flagging:**
- **Weak chain:** prereq mastery ≥ 70% AND dependent avg < 50% — the dependency is questionable
- **Strong chain:** prereq mastery ≥ 70% AND dependent avg ≥ 70% — the pairing holds

### 3. Ebbinghaus Forgetting Curve — Mastery Decay (`mastery_tracker.py`)

Hermann Ebbinghaus (1885) showed that recall after time `t` decays roughly exponentially. The modern formulation used here:

$$R(d) = \max(F, S \cdot e^{-\lambda d})$$

where:
- `S` = raw mastery score at last assessment (0–100),
- `d` = days since assessment,
- `λ` = `MASTERY_DECAY_LAMBDA = 0.02` (tuned so a threshold-level skill halves in ~35 days),
- `F` = `MASTERY_DECAY_FLOOR = 20.0` — the stable "never-fully-forgotten" residual, consistent with findings that well-encoded material plateaus rather than decaying to zero.

**Why use decayed scores, not raw?** The path adapter, scope checker, and review queue all consume `decayed_score`, not `mastery_score`. A 95% mastered three months ago should not block remediation — the decayed value surfaces the stale memory trace honestly. A `needs_review` flag fires whenever `raw ≥ THRESHOLD AND decayed < THRESHOLD`.

### 4. Spaced Repetition Scheduler (`review_scheduler.py`)

Setting `R(d) = T` (the mastery threshold, 85%) in the decay equation and solving for `d` gives the **optimal review interval** — the latest point at which the learner's retention will still be above threshold:

$$d = \frac{\ln(S / T)}{\lambda}$$

Implications:
- A skill at exactly the threshold (`S = T`) has interval `d = 0` — review now.
- A skill above threshold gets a longer interval in proportion to how far above it sits (the `ln` shape matches the intuition from the SM-2 family: every successful review roughly multiplies the interval).
- A skill far above threshold (e.g., 98%) compounds quickly — well-learned material stays dormant longer.

The scheduler emits a **review queue** with per-skill records:

| Field | Meaning |
|---|---|
| `interval_days` | Computed interval until next review |
| `next_review_at` | ISO timestamp of the next scheduled review |
| `days_until_review` | Signed offset — negative means overdue |
| `urgency` | `overdue` / `due_today` / `due_this_week` / `scheduled` |

Records are sorted by `days_until_review` ascending so the most overdue skills surface first. Surfaced via `GET /api/review-schedule/{path_id}`.

### 5. Bloom's Taxonomy + Zone of Proximal Development (`question_generator.py`)

**Bloom's Taxonomy** (Bloom 1956, revised by Anderson & Krathwohl 2001) partitions cognitive learning objectives into six ascending levels:

| Level | Verb family | Example |
|---|---|---|
| Remember | list, define, recall | *Name the three phases of the RAG pipeline.* |
| Understand | explain, summarise, classify | *Why does the scope gate use cosine similarity and not Euclidean distance?* |
| Apply | use, execute, implement | *Write a function that retrieves top-5 chunks from a FAISS index.* |
| Analyze | compare, differentiate, attribute | *Compare the memory cost of full-FT versus QLoRA for a 4B model.* |
| Evaluate | critique, justify, defend | *Is a 4-bit critic acceptable for a medical RAG system? Defend your answer.* |
| Create | design, compose, construct | *Design a 3-session remediation plan for a student weak in Docker and SQL.* |

**Vygotsky's ZPD** (Vygotsky 1978) is the band of tasks that a learner cannot yet solve alone but *can* solve with scaffolding. Tasks below ZPD bore; tasks above block. Optimal instruction targets **one step above current competence**.

Combining the two, the generator maps mastery to a two-level Bloom window:

| Current mastery | Target Bloom window | Rationale |
|---|---|---|
| < 40% | remember, understand | build foundation |
| 40–70% | understand, apply | consolidate |
| 70–85% | apply, analyze | extend |
| ≥ 85% (beginner/intermediate session) | analyze, evaluate | push into higher-order reasoning |
| ≥ 85% (advanced session) | evaluate, create | open-ended synthesis |

The server computes the average *decayed* mastery for a session's skills and passes it to `generate_session_quiz(..., current_mastery=...)`. The prompt used by every `critic.generate_questions` implementation is augmented with a Bloom-hint block targeting that window, and every returned `QuizQuestion` carries a `bloom_level` label that the UI can surface as a cognitive badge.

### 6. Knowledge-Transfer Metrics (`knowledge_transfer.py`)

A core question in curriculum design: **does mastering a prerequisite actually translate into success on the downstream session?** This formalises *transfer of learning* (Thorndike 1901; Perkins & Salomon 1992).

**Pearson correlation.** For each session with at least one quiz attempt, we pair `(avg_prereq_mastery, session_score)` and compute:

$$r = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sqrt{\sum_i (x_i - \bar x)^2} \sqrt{\sum_i (y_i - \bar y)^2}}$$

`r ≈ 1` means the graph's prereq structure predicts outcomes well; `r ≈ 0` means mastery of prereqs says nothing about downstream performance. Only reported when n ≥ 3 pairs (otherwise statistically meaningless).

**Per-prerequisite transfer strength.** For each prerequisite skill X:

$$\text{transfer}(X) = \min(\text{mastery}(X), \text{avg dependent score})$$

The `min` captures the idea that *a skill has only transferred as far as the weakest link in its downstream chain*. A skill with 95% mastery whose dependents average only 40% has a transfer score of 40% — a red flag.

**Automatic flagging:**
- **Weak chain:** prereq mastery ≥ 70% AND dependent avg < 50% — the dependency is questionable
- **Strong chain:** prereq mastery ≥ 70% AND dependent avg ≥ 70% — the pairing holds

### 7. Metacognitive Calibration — Dunning-Kruger Signal (`/api/calibration`)

Flavell (1979) defined **metacognition** as "knowing what you know." Kruger & Dunning (1999) showed that unskilled individuals systematically overestimate competence, while experts often underestimate it. Well-calibrated self-assessment is a leading indicator of expert learning behaviour.

Before submitting a quiz, the student provides a `self_confidence` rating (0–100). After grading, we compute the **confidence-performance gap**:

$$\text{gap}_i = \text{confidence}_i - \text{actual}_i$$

Aggregates exposed by `/api/calibration/{path_id}`:
- **mean_gap** — systematic over/under-confidence (signed)
- **mean_abs_gap** — overall calibration error
- **correlation (Pearson r)** — do confidence and performance move together across attempts?
- **pattern** — `overconfident` (mean_gap > 10), `underconfident` (< -10), or `calibrated`

A learner with `r → 1` and `mean_abs_gap → 0` has accurate metacognition — they know which gaps they have. This signal feeds future work on adaptive remediation: overconfident students need a reality-check loop; underconfident ones need encouragement on questions they actually mastered.

### 8. Cognitive Load Theory — Per-Session Friction (`/api/cognitive-load`)

Sweller (1988) decomposed mental effort during learning into three components:

| Type | Source | Example |
|---|---|---|
| **Intrinsic** | Inherent complexity of the material | Recursion is harder than `if` statements |
| **Extraneous** | Unnecessary load from poor instructional design | Confusing UI, missing context, broken examples |
| **Germane** | Productive load that builds schemas | Practice, worked examples |

We approximate load from two observable signals per session:
- **doubt_query_count** — number of `/session/.../ask` calls (high → learner confused, proxy for intrinsic + extraneous load)
- **study_duration_seconds** — wall-clock time between `start` and `complete` (high → either deep engagement *or* struggle)

Each signal is normalised against the per-path max and combined:

$$\text{load\_score} = 100 \cdot (0.5 \cdot \widehat{\text{doubts}} + 0.5 \cdot \widehat{\text{duration}})$$

A `load_level` of `high` (≥70), `moderate` (40–69), or `low` (<40) flags sessions that warrant instructional redesign — the content, not the student, may be the bottleneck.

### 9. Chunk-Level Citation Provenance (`pipeline._build_citations`)

A known failure mode of RAG systems is *faithfulness drift* — the LLM rephrases or hallucinates even when the retrieved excerpt is correct (Gao et al. 2023 on "RAG hallucinations"). The cure is **traceability**: every claim must resolve back to a specific chunk.

For every document-grounded answer, `QueryResult.metadata.citations` now contains, for each chunk that passed the similarity gate:

| Field | Use |
|---|---|
| `chunk_index` | Position in retrieval result (0 = best match) |
| `similarity_score` | Cosine similarity to query |
| `source_file` / `source_type` | File name and type (pdf, web, transcript, guide) |
| `page` / `resource_name` / `url` | Stable locator — exact page or URL |
| `preview` | First 240 characters of the chunk |

This is the substrate for an inline-citation UI (planned) and for automated faithfulness audits: given an answer sentence, we can score it against its cited chunks with the critic's `evaluate_answer` to detect drift.

---

## Fine-Tuning the SLM — Theory & Practice

This project includes a complete **knowledge-distillation and fine-tuning pipeline** to train Gemma 3 4B as a specialised RAG validation critic. This section explains both the theoretical foundations and practical steps.

### Theoretical Background

#### 1. Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) transfers knowledge from a large **teacher** model to a smaller **student** model. In this system:

- **Teacher:** Google Gemini 2.5 Flash (cloud LLM) — produces high-quality validation judgements
- **Student:** Gemma 3 4B (local SLM) — learns to replicate those judgements

The teacher labels `(question, excerpt)` pairs with structured JSON containing confidence scores, relevance flags, answer types, and explanations. The student then learns to produce the same output format and scoring behaviour through supervised fine-tuning on these labels.

#### 2. QLoRA (Quantised Low-Rank Adaptation)

Full fine-tuning of a 4B-parameter model requires ~32 GB of GPU memory. **QLoRA** (Dettmers et al., 2023) makes this feasible on consumer GPUs by combining two techniques:

**Quantisation (4-bit NormalFloat):**
The base model is loaded in 4-bit precision using the NF4 data type, reducing memory from ~8 GB (FP16) to ~2.5 GB. The NF4 format is information-theoretically optimal for normally-distributed weights:

$$w_{quantised} = \text{NF4}(w_{fp16})$$

**Double quantisation** further compresses the quantisation constants themselves, saving an additional ~0.4 GB.

**Low-Rank Adaptation (LoRA):**
Instead of updating all 4B parameters, LoRA (Hu et al., 2021) injects small trainable matrices into each attention layer. For a weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA decomposes the update as:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.

In this project: $r = 16$, applied to all attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward layers (`gate_proj`, `up_proj`, `down_proj`). This means only **~1–2%** of the total parameters are trainable (~40M out of 4B), reducing GPU memory for gradients and optimizer states from ~24 GB to ~1.5 GB.

**Combined memory footprint:**

| Component | Memory |
|---|---|
| Base model (NF4 quantised) | ~2.5 GB |
| LoRA adapters (trainable) | ~0.1 GB |
| Gradients + optimizer states | ~1.5 GB |
| Activations (batch=2, seq=1024) | ~1.5 GB |
| **Total** | **~6 GB** |

This fits comfortably on a 6 GB GPU (e.g., RTX 4050 Laptop).

#### 3. Supervised Fine-Tuning (SFT) with Chat Templates

The training uses **SFTTrainer** from HuggingFace TRL, which formats each example as a multi-turn conversation using Gemma's chat template:

```
<start_of_turn>user
[System prompt + question + excerpt]
<end_of_turn>
<start_of_turn>model
{"confidence": 88, "is_relevant": true, "answer_type": "document", ...}
<end_of_turn>
```

The model learns to generate the JSON completion given the structured prompt. The loss is computed only on the model's response tokens (not the prompt), so the model learns what to *output* rather than memorising the input.

#### 4. Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 3 | Enough for convergence on ~310 examples without overfitting |
| Batch size | 2 | Fits 6 GB VRAM |
| Gradient accumulation | 4 | Effective batch size = 8 |
| Learning rate | 2e-4 | Standard for QLoRA |
| LR scheduler | Cosine | Smooth decay prevents catastrophic forgetting |
| Warmup steps | 5 | Brief ramp-up for training stability |
| Max gradient norm | 0.3 | Clips extreme gradients to prevent instability |
| Optimizer | AdamW 8-bit | Memory-efficient variant via bitsandbytes |
| Weight decay | 0.01 | Light regularisation |
| LoRA rank (r) | 16 | Balance between capacity and efficiency |
| LoRA alpha | 16 | Scaling factor (alpha/r = 1.0) |
| LoRA dropout | 0.0 | No dropout — dataset is small enough that we want full learning capacity |
| Max sequence length | 1024 | Sufficient for prompt + JSON response |

#### 5. Synthetic Data Generation & Confidence Calibration

A key challenge was **confidence calibration** — the model must produce confidence scores that align with the system's `CONFIDENCE_THRESHOLD = 85`. If training data is biased toward low confidence, the model will never trigger "document" answers.

**The calibration problem (v1):** The initial synthetic data generator produced a training set where only 8% of examples had confidence ≥ 85 (mean = 52.8). The trained model's confidence scores capped at 83 and never reached the threshold — resulting in 0% document answer rate.

**The fix (v2):** The `assign_confidence()` heuristic was redesigned to weight similarity more heavily (`0.7 × similarity + 0.3 × keyword_overlap`) and map the high band (combined ≥ 0.65) directly to the 85–100 range:

```python
# v2 calibration
combined = similarity * 0.7 + keyword_overlap * 0.3

if combined >= 0.65:       # High band → 85-100 (triggers "document" answers)
    base = int(85 + (combined - 0.65) * 43)
    conf = min(100, max(85, base + random.randint(-3, 3)))
elif combined >= 0.50:     # Medium-high → 60-84
    ...
elif combined >= 0.35:     # Medium-low → 35-59
    ...
```

**Training data distribution comparison:**

| Band | v1 (broken) | v2 (calibrated) |
|---|---|---|
| ≥ 85 (document) | 8% | 55% |
| 60–84 | 31% | 21% |
| 35–59 | — | 2.5% |
| < 35 | 37% | 21% |
| **Mean confidence** | 52.8 | 69.3 |
| **Max confidence** | 93 | 97 |

#### 6. GGUF Conversion & Quantisation Pipeline

After training, the LoRA adapter must be converted to a format Ollama understands. The pipeline has three stages:

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  merge_lora_cpu   │ →  │  convert_hf_to   │ →  │  ollama create   │
│  (LoRA + Base     │    │  _gguf.py (F16)  │    │  --quantize      │
│   → clean FP16)   │    │  (7.76 GB GGUF)  │    │  q4_K_M (2.5GB) │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Why this 3-step process?**

1. **CPU merge** (`merge_lora_cpu.py`) — The QLoRA adapter was trained with 4-bit quantisation. Merging LoRA on the GPU produces weights with bitsandbytes quantisation metadata that llama.cpp cannot parse. Loading the base model on CPU in FP16 and applying the LoRA produces clean, standard FP16 weights.

2. **HF → GGUF** (`convert_hf_to_gguf.py` from llama.cpp) — Converts the PyTorch safetensors to GGUF format, which is llama.cpp's native tensor format. The output is an FP16 GGUF file (~7.76 GB).

3. **Ollama quantisation** (`ollama create --quantize q4_K_M`) — Quantises the FP16 GGUF to Q4_K_M (4-bit with K-quant mixing), reducing the file to ~2.5 GB while preserving most accuracy. Q4_K_M uses higher precision for attention layers and lower for feed-forward layers.

### Practical Steps

#### Step 1 — Generate training data

**Option A: Gemini-labelled (teacher distillation)**
```bash
python generate_training_data.py                  # 50 seed questions × top-k chunks
python generate_training_data.py --expand          # add paraphrased augmentation
```

**Option B: Synthetic (no API calls)**
```bash
python generate_synthetic_data.py                              # default: training_data.jsonl
python generate_synthetic_data.py --out training_data_v2.jsonl  # custom output path
```

The synthetic generator uses FAISS similarity + keyword overlap heuristics. It produces ~285 examples from 75 seed questions (55 in-scope + 20 out-of-scope) plus 20 augmented paraphrases.

#### Step 2 — Fine-tune with QLoRA

```bash
pip install -r requirements-finetune.txt

# Default training
python finetune.py --data training_data_v2.jsonl --output-dir finetuned_model_v2

# Custom hyperparameters
python finetune.py --data training_data_v2.jsonl --output-dir finetuned_model_v2 \
    --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4

# Resume from checkpoint
python finetune.py --data training_data_v2.jsonl --output-dir finetuned_model_v2 --resume
```

Requires an **NVIDIA GPU with ≥ 6 GB VRAM** (RTX 3060 / 4050 / 4060 or better). Training takes ~35 minutes on 310 examples (3 epochs).

#### Step 3 — Merge LoRA on CPU (clean FP16)

```bash
python merge_lora_cpu.py --lora finetuned_model_v2/lora_adapter --out finetuned_model_v2/merged_fp16
```

This loads the base model on CPU (~8 GB RAM) and produces a clean FP16 model that llama.cpp can convert.

#### Step 4 — Convert to GGUF

```bash
# Requires llama.cpp cloned locally
python /path/to/llama.cpp/convert_hf_to_gguf.py finetuned_model_v2/merged_fp16 \
    --outfile finetuned_model_v2/gguf/gemma3-critic-v2-f16.gguf --outtype f16
```

#### Step 5 — Deploy to Ollama

Create a Modelfile pointing to the GGUF, then import with quantisation:

```bash
ollama create gemma3-critic-v2 -f finetuned_model_v2/Modelfile_gguf --quantize q4_K_M
```

Then update `config.py`:

```python
OLLAMA_MODEL_NAME = "gemma3-critic-v2"
```

#### Step 6 — Evaluate

```bash
python evaluate.py                                 # compare all available backends
python evaluate.py --backends gemini gemma          # specific pair
python evaluate.py --out results.json               # save raw per-question data
```

---

## Evaluation Results

### Current Production Model — v3-new (`gemma3-critic-v3-new`)

The production critic is the third-generation fine-tune, retrained on an expanded 300-question bank with refined confidence calibration and JSON formatting. The results below are from `evaluate.py` on the latest held-out evaluation set.

| Metric | gemma3-critic-v3-new | Interpretation |
|---|---|---|
| JSON validity | **100.0%** | Every response parses cleanly |
| Avg confidence (in-scope) | **89.4%** | Comfortably above the 85% confidence gate |
| Avg confidence (out-scope) | **0.0%** | Sharp out-of-scope discrimination |
| Scope accuracy | **100.0%** | Matches the ground-truth scope label on every question |
| Document answer rate | **100.0%** | Never falls back to general knowledge on in-scope questions |
| Avg latency | 5.34s | Consistent local inference — no rate limits |
| P50 latency | 5.26s | Low jitter |
| Error rate | 0.0% | Zero failed completions |

### Iteration History — v1 → v2 → v3

The critic model evolved across three training iterations. The v1 → v2 jump solved a confidence-calibration failure; v3 improved scope discrimination and stability.

| Iteration | Avg Conf (in-scope) | Avg Conf (out-scope) | Doc Answer Rate | Scope Accuracy | Notes |
|---|---|---|---|---|---|
| v1 (`gemma3-critic`) | 67.0 | 66.0 | 0% | 80% | Confidence capped at 83 — never triggered document answers |
| v2 (`gemma3-critic-v2`) | 87.5 | 71.8 | 100% | 80% | Calibrated confidence distribution; over-confident on OOS |
| **v3-new (`gemma3-critic-v3-new`)** | **89.4** | **0.0** | **100%** | **100%** | Current production model — sharp OOS rejection restored |

**Key takeaways:**
- v3-new **combines the strengths of both prior iterations**: v2's high in-scope confidence with near-zero out-of-scope confidence (a trait previously only achievable with Gemini cloud)
- 100% scope accuracy means the model now matches the ground-truth in/out-of-scope label on every held-out question
- Latency is stable at ~5s locally with no network variance, compared to Gemini's 6–156s (rate-limited)

### When to Use Which Backend

| Scenario | Best Choice | Why |
|---|---|---|
| Production (document-grounded answers) | **Gemma v3-new** | 100% document answer rate on in-scope, 0% confidence on out-of-scope |
| Offline / no internet | **Gemma v3-new** | Fully local via Ollama |
| Low latency / predictable tail | **Gemma v3-new** | ~5s local vs Gemini's variable 2–157s under rate limits |
| Cost-sensitive deployment | **Gemma v3-new** | Zero API cost |
| Cloud / zero-infrastructure setup | **Gemini 2.5 Flash** | No Ollama/GPU required; automatic fallback chain on rate limits |

---

## Web UI Features

### Landing Page
- Three.js animated gradient mesh background (128x128 undulating surface with layered sine waves)
- Floating ambient orbs with pulsing glow
- Mouse-reactive camera sway
- Feature cards explaining the system architecture

### Authentication
- JWT tokens stored in localStorage (24-hour expiry)
- bcrypt password hashing (direct, without passlib)
- Route guards redirect unauthenticated users to login

### Chatbot (`/chat`)
- Real-time question/answer with typing animation
- Source badges: **Document** (green), **General Knowledge** (amber), **Out of Scope** (red)
- Similarity and confidence scores displayed per answer
- Suggestion chips for quick starter questions
- Persistent chat history per user (SQLite)

### Learning Pipeline (`/learn`)
- **Sidebar** — path selector dropdown, session list with status icons (not started / in progress / completed), mastery and adapt buttons
- **Path validation panel** — displays auto-correction log after loading a path
- **Study panel** — chat-style Q&A interface against session-specific materials with source badges; Materials button opens a modal showing all ingested sources grouped by type (PDFs, web resources with URLs, built-in guides with content previews, video resources with status badges and direct links)
- **Quiz panel** — MCQ (radio buttons) and open-ended (text areas) questions with source labels and reference text blocks (code snippets as `<pre>`, passages as `<blockquote>`); grading results show letter grade, per-skill scores, and detailed per-question feedback
- **Mastery dashboard** — overall percentage, mastered/review/weak counts, per-skill progress bars colour-coded by status
- **Adaptation panel** — displays human-readable adaptation log (sessions compressed, remediation injected, resources suggested)

---

## API Endpoints

### Authentication & Chat (Original)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/signup` | No | Register a new user |
| `POST` | `/api/auth/login` | No | Authenticate and receive JWT |
| `GET` | `/api/auth/me` | Yes | Get current user info |
| `POST` | `/api/chat` | Yes | Send a question to the RAG pipeline |
| `GET` | `/api/chat/history` | Yes | Retrieve user's chat history |
| `GET` | `/api/health` | No | Server and pipeline status |

### Learning Path Pipeline

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/path/available` | Yes | List available learning path JSON files |
| `POST` | `/api/path/load` | Yes | Load a learning path JSON into the database |
| `POST` | `/api/path/{id}/validate` | Yes | Validate and auto-correct a loaded path |
| `GET` | `/api/path/{id}/sessions` | Yes | List sessions with progress status |
| `POST` | `/api/session/{path_id}/{n}/start` | Yes | Ingest content and start a study session |
| `POST` | `/api/session/{path_id}/{n}/ask` | Yes | Ask a question in study mode (session-scoped RAG) |
| `POST` | `/api/session/{path_id}/{n}/complete` | Yes | Mark a session as completed |
| `GET` | `/api/session/{path_id}/{n}/materials` | Yes | Get ingested source details (PDFs, web, guides, transcripts) |
| `POST` | `/api/quiz/{path_id}/{n}/generate` | Yes | Generate a quiz for a session |
| `POST` | `/api/quiz/{path_id}/{n}/submit` | Yes | Submit quiz answers for grading |
| `GET` | `/api/quiz/{path_id}/{n}/results` | Yes | Get quiz attempt history |
| `GET` | `/api/mastery/{path_id}` | Yes | Get mastery summary for a path (with decayed scores + `needs_review` flags) |
| `GET` | `/api/review-schedule/{path_id}` | Yes | Ebbinghaus-derived spaced-repetition queue (urgency per skill) |
| `GET` | `/api/transfer/{path_id}` | Yes | Knowledge-transfer metrics (Pearson correlation + per-prereq strength) |
| `GET` | `/api/calibration/{path_id}` | Yes | Metacognitive calibration — confidence vs. actual performance (Dunning-Kruger signal) |
| `GET` | `/api/cognitive-load/{path_id}` | Yes | Per-session cognitive-load signals (doubt count + study duration) |
| `POST` | `/api/path/{id}/adapt` | Yes | Generate an adapted learning path (uses decayed mastery + graph-constrained reorder) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | BAAI/bge-base-en-v1.5 via SentenceTransformers |
| Vector Store | FAISS (L2 index, local persistence) — global + per-session indexes |
| LLM Critic | Google Gemini 2.5 Flash + fallback chain |
| SLM Critic | Gemma 3 4B fine-tuned via QLoRA, deployed via Ollama (Q4_K_M) |
| SLM Base | Gemma 3 4B (base) via Ollama — all generation tasks |
| Fine-Tuning | PEFT + BitsAndBytes (NF4) + TRL SFTTrainer |
| GGUF Conversion | llama.cpp (`convert_hf_to_gguf.py`) + CPU LoRA merge |
| PDF Parsing | PyPDFLoader (LangChain) |
| Chunking | RecursiveCharacterTextSplitter |
| Web Scraping | trafilatura (primary) + BeautifulSoup (fallback) |
| Search | DDGS / Brave Search / DuckDuckGo HTML (3-tier fallback for resource URL resolution) |
| Video Transcripts | yt-dlp + OpenAI Whisper (optional — graceful degradation with direct links) |
| Backend | FastAPI + Uvicorn |
| Auth | JWT (python-jose) + bcrypt |
| Database | SQLite (WAL mode, 7 tables) |
| Frontend | HTML/CSS/JS, Three.js (ES modules via CDN) |
| CLI | argparse |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No PDF files found` | Place `.pdf` files inside `Dataset/` |
| Gemini 429 / ResourceExhausted | Automatic retry with fallback chain is built in; wait if persistent |
| Ollama connection refused | Ensure Ollama is running: `ollama serve` or check tray icon |
| `Unknown CRITIC_BACKEND` | Set to `"gemini"`, `"gemma"`, or `"mock"` in `config.py` |
| Windows encoding crash | `server.py` and `main.py` wrap stdout in UTF-8 |
| Weak retrieval scores | Run `python main.py --rebuild` or tune `SIMILARITY_THRESHOLD` |
| Auth errors / expired token | Log out and log back in; tokens expire after 24 hours |
| Port 8000 in use | `uvicorn server:app --port 8001` |
| Fine-tuning OOM | Reduce `--batch-size` to 1 or `--grad-accum` to 8 |
| GGUF conversion fails | Use `merge_lora_cpu.py` first for clean FP16, then convert |
| Ollama model assertion error | Ensure GGUF was converted from the CPU-merged FP16, not the 4-bit merged model |
| Path validation slow | Base model (`gemma3:4b`) must be running in Ollama — validation checks each session via LLM |
| Quiz generation fails | Ensure per-session FAISS index exists — run study mode first to trigger ingestion |
| Web scraping blocked | The system uses 3-tier search (DDGS → Brave → DDG HTML) and 2-tier extraction (trafilatura → BS4) with graceful fallback to JSON guides. Video URLs are auto-detected and shown as direct links if yt-dlp is not installed |
| No learning path files found | Place `.json` files inside `Learning Path Inputs/` |

---

## Security

- `.env` is gitignored — API keys never reach the repository
- `users.db` is gitignored — user data stays local
- `vectorstore/` and `session_vectorstores/` are gitignored — regeneratable from source content
- `finetuned_model*/` is gitignored — model weights stay local (available on HuggingFace, see below)
- Passwords are hashed with bcrypt (never stored in plain text)
- JWT tokens are signed with a configurable secret key
- All learning pipeline API endpoints require JWT authentication
- Original learning path JSONs are never modified — corrections saved as separate files

---

## Future Improvements

- Ensemble critic (Gemma for speed + Gemini for verification on borderline cases)
- Raise similarity threshold to 0.40 to improve out-of-scope filtering
- Scale training data to 1000+ examples with more diverse out-of-scope questions
- Multi-round fine-tuning with DPO (Direct Preference Optimisation)
- Full-text faithfulness audit UI (per-sentence chunk attribution overlay on the study-chat answer)
- WebSocket streaming for real-time token-by-token responses
- User file upload through the web UI
- Automated faithfulness audit — run `critic.evaluate_answer` on every answer sentence against its cited chunks, surface drift as a confidence penalty
- Adaptive remediation tied to calibration pattern (overconfident → reality-check loops; underconfident → positive reinforcement on solved gaps)
- Collaborative learning paths (multi-user progress tracking)
- Export mastery reports as PDF

---

## Pre-Trained Models (HuggingFace)

All fine-tuned Gemma 3 4B critic iterations are available on HuggingFace. **v3 is the current production model** — the earlier iterations are retained for reproducibility of the training history documented above.

| Model | Status | Description | Link |
|---|---|---|---|
| **gemma3-critic-v3** | **Current** | Expanded 300-question bank, calibrated confidence, sharp scope discrimination (100% scope accuracy) | [V3gito/gemma3-critic-v3](https://huggingface.co/V3gito/gemma3-critic-v3) |
| gemma3-critic-v2 | Archived | Calibrated confidence distribution — 100% document answer rate but over-confident on out-of-scope | [V3gito/gemma3-critic-v2](https://huggingface.co/V3gito/gemma3-critic-v2) |
| gemma3-critic-v1 | Archived | First fine-tune — confidence capped at 83, 0% document answer rate (motivating example) | [V3gito/gemma3-critic-v1](https://huggingface.co/V3gito/gemma3-critic-v1) |

Each repo contains the GGUF file (F16) and LoRA adapter checkpoints. To use the current model with Ollama:

```bash
# Download the v3 GGUF from HuggingFace, then create an Ollama model
ollama create gemma3-critic-v3-new -f Modelfile_gguf --quantize q4_K_M
```

The CLI name `gemma3-critic-v3-new` matches `OLLAMA_MODEL_NAME` in `config.py`, so no further configuration is needed.

---

## References

### Retrieval-Augmented Generation & Distillation
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401
- Gao, Y., et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531

### Efficient Fine-Tuning
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models.* arXiv:2305.14314

### Cognitive Science of Learning
- Ebbinghaus, H. (1885). *Über das Gedächtnis* (On Memory). Leipzig: Duncker & Humblot. — exponential forgetting curve
- Bloom, B. S. (1956). *Taxonomy of Educational Objectives, Handbook I: Cognitive Domain.* Longmans.
- Anderson, L. W., & Krathwohl, D. R. (Eds.) (2001). *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy.* Longman.
- Vygotsky, L. S. (1978). *Mind in Society: The Development of Higher Psychological Processes.* — Zone of Proximal Development
- Sweller, J. (1988). *Cognitive Load During Problem Solving: Effects on Learning.* Cognitive Science, 12(2), 257–285.
- Flavell, J. H. (1979). *Metacognition and Cognitive Monitoring: A New Area of Cognitive-Developmental Inquiry.* American Psychologist, 34(10), 906–911.
- Kruger, J., & Dunning, D. (1999). *Unskilled and Unaware of It: How Difficulties in Recognizing One's Own Incompetence Lead to Inflated Self-Assessments.* Journal of Personality and Social Psychology, 77(6), 1121–1134.
- Thorndike, E. L. (1901). *The Influence of Improvement in One Mental Function upon the Efficiency of Other Functions.* Psychological Review, 8(3), 247–261. — transfer of learning
- Perkins, D. N., & Salomon, G. (1992). *Transfer of Learning.* International Encyclopedia of Education.

### Standards & Formal Methods
- European Commission. *ESCO — European Skills, Competences, Qualifications and Occupations.* ec.europa.eu/esco — skill taxonomy used in session skill-details mapping
- Kahn, A. B. (1962). *Topological Sorting of Large Networks.* Communications of the ACM, 5(11), 558–562.

---

## License

No license file is included yet. Add a `LICENSE` file before open-source distribution.
