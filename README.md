# Self-Correcting RAG System (Actor-Critic)

A full-stack Retrieval-Augmented Generation system that answers questions from PDF documents, validates answers using a swappable Critic model (cloud LLM **or** local SLM), and serves everything through a professional web interface with user authentication.

---

## Key Features

- **Actor-Critic RAG pipeline** — retrieves document chunks, then validates them with a Critic model before responding
- **LLM / SLM selection** — switch between Google Gemini 2.5 Flash (cloud LLM) and Gemma 3 4B (local SLM via Ollama) with a single config change
- **Three answer modes** — exact document excerpt, general-knowledge fallback, or out-of-scope rejection
- **Web UI** — dark-themed glassmorphism design with Three.js animated gradient-mesh background
- **Authentication** — JWT-based signup/login with bcrypt password hashing and per-user chat history
- **Two interfaces** — browser-based chatbot (FastAPI) and terminal CLI
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
                                    confidence ≥ 85%              confidence < 85%
                                    Return document               LLM/SLM fallback
                                    excerpt verbatim              from general knowledge
```

### Module Breakdown

| Module | Role |
|---|---|
| `config.py` | All tuneable parameters: paths, models, thresholds, backend selection |
| `ingestion.py` | Discovers PDFs in `Dataset/`, loads pages with PyPDFLoader, splits into overlapping chunks |
| `vector_store.py` | Embeds chunks with BGE-base-en-v1.5 (768d), builds/loads FAISS index, applies scope gate |
| `critic.py` | `GeminiCritic` (cloud), `GemmaCritic` (local Ollama), `MockCritic` (offline), and `create_critic()` factory |
| `pipeline.py` | Orchestrates Actor → Scope Gate → Critic → Confidence Gate flow |
| `server.py` | FastAPI backend — auth endpoints, chat API, static file serving, pipeline startup |
| `auth.py` | JWT token creation/verification, bcrypt password hashing |
| `database.py` | SQLite user store + chat history persistence |
| `main.py` | Terminal CLI with interactive loop (alternative to web UI) |

### Frontend

| File | Purpose |
|---|---|
| `static/index.html` | Landing page with feature cards and hero section |
| `static/login.html` | Login form with validation and error display |
| `static/signup.html` | Registration form with password confirmation |
| `static/chat.html` | Chatbot interface with typing indicators, source badges, suggestion chips, chat history |
| `static/js/background.js` | Three.js animated gradient mesh — undulating surface with layered sine waves, floating orbs, mouse-reactive camera |
| `static/js/auth.js` | Token management, authenticated fetch wrapper, route guards |
| `static/css/style.css` | Dark theme with CSS variables, glassmorphism cards, responsive layout, animations |

---

## Project Structure

```text
Code/
├─ Dataset/
│  ├─ Principles-of-Data-Science-WEB.pdf
│  └─ Introduction_to_Python_Programming_-_WEB.pdf
├─ static/
│  ├─ css/
│  │  └─ style.css
│  ├─ js/
│  │  ├─ auth.js
│  │  └─ background.js
│  ├─ index.html
│  ├─ login.html
│  ├─ signup.html
│  └─ chat.html
├─ vectorstore/          # generated FAISS index (gitignored)
├─ users.db              # SQLite user database (gitignored)
├─ config.py
├─ ingestion.py
├─ vector_store.py
├─ critic.py
├─ pipeline.py
├─ server.py
├─ auth.py
├─ database.py
├─ main.py
├─ requirements.txt
├─ .env                  # API keys (gitignored)
└─ README.md
```

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- **For Gemini backend:** Internet access + Gemini API key
- **For Gemma backend:** [Ollama](https://ollama.com) installed locally

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
CRITIC_BACKEND = "gemini"    # or "gemma"
```

#### Option A — Gemini (cloud LLM)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Option B — Gemma 3 4B (local SLM)

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the model:
   ```bash
   ollama pull gemma3:4b
   ```
3. Ensure Ollama is running (`ollama serve` or the system tray app)
4. Set `CRITIC_BACKEND = "gemma"` in `config.py`

No API key or internet is needed for inference once the model is downloaded.

---

## Run

### Web UI (recommended)

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

- **/** — Landing page
- **/signup** — Create an account
- **/login** — Sign in
- **/chat** — Chatbot (requires login)

### CLI mode

```bash
python main.py               # interactive terminal loop
python main.py --rebuild     # force-rebuild the FAISS index
python main.py --mock        # use MockCritic (no API key needed)
python main.py --threshold 0.25 --confidence 90   # override gates
```

---

## Configuration

All parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CRITIC_BACKEND` | `"gemini"` | `"gemini"` for cloud LLM, `"gemma"` for local SLM, `"mock"` for offline |
| `CHUNK_SIZE` | 1000 | Characters per document chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model (768 dims) |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.20 | Cosine similarity floor for scope gate |
| `CONFIDENCE_THRESHOLD` | 85 | Critic confidence floor (0–100) |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash` | Cloud LLM model name |
| `OLLAMA_MODEL_NAME` | `gemma3:4b` | Local SLM model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |

**Tuning tips:**
- Lower `SIMILARITY_THRESHOLD` if relevant questions are marked out-of-scope
- Increase `CONFIDENCE_THRESHOLD` for stricter document-only answers
- Increase `TOP_K` if answers seem incomplete

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

## Critic Backend Details

### Gemini (Cloud LLM)

- **Model:** Google Gemini 2.5 Flash
- **Pros:** High accuracy, strong instruction-following, fast
- **Cons:** Requires API key, internet, subject to rate limits (automatic retry built in)

### Gemma 3 4B (Local SLM)

- **Model:** Gemma 3 4B running via Ollama
- **Pros:** Fully offline, no API costs, complete data privacy
- **Cons:** Requires ~3.3 GB disk space, slower on CPU-only machines

Both backends share identical validation and fallback prompts. The `create_critic()` factory in `critic.py` reads `CRITIC_BACKEND` from config and returns the appropriate implementation.

---

## Web UI Features

### Landing Page
- Three.js animated gradient mesh background (128×128 undulating surface with layered sine waves)
- Floating ambient orbs with pulsing glow
- Mouse-reactive camera sway
- Feature cards explaining the system architecture

### Authentication
- JWT tokens stored in localStorage (24-hour expiry)
- bcrypt password hashing (direct, without passlib)
- Route guards redirect unauthenticated users to login

### Chatbot
- Real-time question/answer with typing animation
- Source badges: **Document** (green), **General Knowledge** (amber), **Out of Scope** (red)
- Similarity and confidence scores displayed per answer
- Suggestion chips for quick starter questions
- Persistent chat history per user (SQLite)

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/signup` | No | Register a new user |
| `POST` | `/api/auth/login` | No | Authenticate and receive JWT |
| `GET` | `/api/auth/me` | Yes | Get current user info |
| `POST` | `/api/chat` | Yes | Send a question to the RAG pipeline |
| `GET` | `/api/chat/history` | Yes | Retrieve user's chat history |
| `GET` | `/api/health` | No | Server and pipeline status |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | BAAI/bge-base-en-v1.5 via SentenceTransformers |
| Vector Store | FAISS (L2 index, local persistence) |
| LLM Critic | Google Gemini 2.5 Flash (cloud) |
| SLM Critic | Gemma 3 4B via Ollama (local) |
| PDF Parsing | PyPDFLoader (LangChain) |
| Chunking | RecursiveCharacterTextSplitter |
| Backend | FastAPI + Uvicorn |
| Auth | JWT (python-jose) + bcrypt |
| Database | SQLite |
| Frontend | HTML/CSS/JS, Three.js (ES modules via CDN) |
| CLI | argparse |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No PDF files found` | Place `.pdf` files inside `Dataset/` |
| Gemini 429 / ResourceExhausted | Automatic retry/backoff is built in; wait if persistent |
| Ollama connection refused | Ensure Ollama is running: `ollama serve` or check tray icon |
| `Unknown CRITIC_BACKEND` | Set `CRITIC_BACKEND` to `"gemini"`, `"gemma"`, or `"mock"` in `config.py` |
| Windows encoding crash | `server.py` and `main.py` wrap stdout in UTF-8 |
| Weak retrieval scores | Run `python main.py --rebuild` or tune `SIMILARITY_THRESHOLD` |
| Auth errors / expired token | Log out and log back in; tokens expire after 24 hours |
| Port 8000 in use | Change port: `uvicorn server:app --port 8001` |

---

## Security

- `.env` is gitignored — API keys never reach the repository
- `users.db` is gitignored — user data stays local
- `vectorstore/` is gitignored — regeneratable from source PDFs
- Passwords are hashed with bcrypt (never stored in plain text)
- JWT tokens are signed with a configurable secret key

---

## Future Improvements

- Citation highlighting at chunk level
- Evaluation suite and benchmark queries
- Support for DOCX / HTML / Markdown ingestion
- WebSocket streaming for real-time token-by-token responses
- User file upload through the web UI

---

## License

No license file is included yet. Add a `LICENSE` file before open-source distribution.
