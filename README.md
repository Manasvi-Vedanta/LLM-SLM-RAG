# Self-Correcting RAG System with Actor-Critic Architecture

A full-stack Retrieval-Augmented Generation system that answers questions from PDF documents, validates answers using a swappable **Critic** model (cloud LLM **or** fine-tuned local SLM), and serves everything through a professional web interface with user authentication.

---

## Key Features

- **Actor-Critic RAG pipeline** — retrieves document chunks, then validates them with a Critic model before responding
- **LLM / SLM selection** — switch between Google Gemini 2.5 Flash (cloud LLM) and a fine-tuned Gemma 3 4B (local SLM via Ollama) with a single config change
- **Complete fine-tuning pipeline** — synthetic data generation, QLoRA training, GGUF conversion, Ollama deployment, and side-by-side evaluation
- **Rate-limit resilient** — Gemini backend cascades through fallback models (`gemini-2.5-flash` → `gemini-3-flash-preview` → `gemini-2.5-flash-lite`) on 429 errors
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
| `config.py` | All tuneable parameters — paths, models, thresholds, backend selection |
| `ingestion.py` | Discovers PDFs in `Dataset/`, loads pages with PyPDFLoader, splits into overlapping chunks |
| `vector_store.py` | Embeds chunks with BGE-base-en-v1.5 (768d), builds/loads FAISS index, applies scope gate |
| `critic.py` | `GeminiCritic` (cloud + fallback chain), `GemmaCritic` (local Ollama), `MockCritic` (offline), `create_critic()` factory |
| `pipeline.py` | Orchestrates Actor → Scope Gate → Critic → Confidence Gate flow |
| `generate_training_data.py` | Gemini-based teacher labelling — runs queries through Gemini to produce labelled JSONL |
| `generate_synthetic_data.py` | Heuristic synthetic data generator — creates training data without API calls using similarity/keyword heuristics |
| `finetune.py` | QLoRA fine-tuning of Gemma 3 4B using PEFT + BitsAndBytes + TRL |
| `merge_lora_cpu.py` | Merges LoRA adapter into base model on CPU — produces clean FP16 weights for GGUF conversion |
| `evaluate.py` | Side-by-side benchmark comparing LLM vs SLM on accuracy, latency, and agreement |
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
├── Dataset/
│   ├── Principles-of-Data-Science-WEB.pdf
│   └── Introduction_to_Python_Programming_-_WEB.pdf
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── auth.js
│   │   └── background.js
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   └── chat.html
├── vectorstore/                  # FAISS index (generated, gitignored)
├── finetuned_model/              # v1 fine-tune artifacts (gitignored)
├── finetuned_model_v2/           # v2 fine-tune artifacts (gitignored)
│   ├── checkpoints/              #   training checkpoints
│   ├── lora_adapter/             #   LoRA adapter weights
│   ├── merged/                   #   4-bit merged model (from finetune.py)
│   ├── merged_fp16/              #   clean FP16 merge (from merge_lora_cpu.py)
│   └── gguf/                     #   GGUF file for Ollama
├── config.py
├── ingestion.py
├── vector_store.py
├── critic.py
├── pipeline.py
├── generate_training_data.py
├── generate_synthetic_data.py
├── finetune.py
├── merge_lora_cpu.py
├── evaluate.py
├── Modelfile
├── server.py
├── auth.py
├── database.py
├── main.py
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

| Route | Page |
|---|---|
| `/` | Landing page |
| `/signup` | Create an account |
| `/login` | Sign in |
| `/chat` | Chatbot (requires login) |

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
| `CRITIC_BACKEND` | `"gemma"` | `"gemini"` for cloud LLM, `"gemma"` for local SLM, `"mock"` for offline |
| `CHUNK_SIZE` | 1000 | Characters per document chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model (768 dims) |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.20 | Cosine similarity floor for scope gate |
| `CONFIDENCE_THRESHOLD` | 85 | Critic confidence floor (0–100) |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash` | Primary cloud LLM model |
| `GEMINI_FALLBACK_MODELS` | `[gemini-3-flash-preview, gemini-2.5-flash-lite]` | Fallback models on rate limit |
| `OLLAMA_MODEL_NAME` | `gemma3-critic-v2` | Local SLM model (fine-tuned) |
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

- **Model:** Google Gemini 2.5 Flash (with automatic fallback to `gemini-3-flash-preview` → `gemini-2.5-flash-lite` on rate limits)
- **Pros:** High accuracy, nuanced scope discrimination, strong instruction-following
- **Cons:** Requires API key, internet, subject to rate limits (automatic retry with exponential backoff built in)

### Gemma 3 4B (Local SLM — Fine-Tuned)

- **Model:** Gemma 3 4B fine-tuned via QLoRA, quantised to Q4_K_M, deployed via Ollama
- **Pros:** Fully offline, no API costs, consistent ~3.5s latency, 100% document answer rate on in-scope questions
- **Cons:** Requires ~2.5 GB disk, less nuanced on out-of-scope boundary questions

Both backends share identical validation and fallback prompts. The `create_critic()` factory in `critic.py` reads `CRITIC_BACKEND` from config and returns the appropriate implementation.

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

### v1 vs v2 Model Comparison

| Metric | Gemma v1 | Gemma v2 | Gemini (cloud) |
|---|---|---|---|
| JSON validity | 100% | 100% | 100% |
| Avg confidence (in-scope) | 67 | **87.8** | 54.1 |
| Max confidence | 83 | **89** | 100 |
| Document answer rate | 0% | **100%** | 25% |
| Avg confidence (out-scope) | 50 | 71.2 | **8.0** |
| Avg latency | **3.74s** | 3.84s | 29.58s |
| Error rate | 0% | 0% | 0% |

**Key takeaways:**
- **v2 solved the core v1 problem** — it now consistently triggers "document" answers for in-scope questions (confidence 85–89)
- **Gemini has better out-of-scope discrimination** — assigns near-zero confidence to irrelevant questions, while Gemma v2 sometimes over-estimates
- **Gemma is 7–40× faster** — consistent ~3.8s vs Gemini's 6–156s (rate-limited)

### When to Use Which

| Scenario | Best Choice | Why |
|---|---|---|
| Production (document-grounded answers) | **Gemma v2** | Only model that consistently triggers "document" answers |
| Strict out-of-scope filtering | **Gemini** | Reliably gives 0 confidence to irrelevant questions |
| Offline / no internet | **Gemma v2** | Fully local via Ollama |
| Low latency | **Gemma v2** | Consistent 3–4s vs Gemini's variable 2–157s |
| Cost-sensitive deployment | **Gemma v2** | Zero API cost |

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
| LLM Critic | Google Gemini 2.5 Flash + fallback chain |
| SLM Critic | Gemma 3 4B fine-tuned via QLoRA, deployed via Ollama (Q4_K_M) |
| Fine-Tuning | PEFT + BitsAndBytes (NF4) + TRL SFTTrainer |
| GGUF Conversion | llama.cpp (`convert_hf_to_gguf.py`) + CPU LoRA merge |
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

---

## Security

- `.env` is gitignored — API keys never reach the repository
- `users.db` is gitignored — user data stays local
- `vectorstore/` is gitignored — regeneratable from source PDFs
- `finetuned_model*/` is gitignored — model weights stay local
- Passwords are hashed with bcrypt (never stored in plain text)
- JWT tokens are signed with a configurable secret key

---

## Future Improvements

- Ensemble critic (Gemma for speed + Gemini for verification on borderline cases)
- Raise similarity threshold to 0.40 to improve out-of-scope filtering
- Scale training data to 1000+ examples with more diverse out-of-scope questions
- Multi-round fine-tuning with DPO (Direct Preference Optimisation)
- Citation highlighting at chunk level
- WebSocket streaming for real-time token-by-token responses
- User file upload through the web UI

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models.* arXiv:2305.14314
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401

---

## License

No license file is included yet. Add a `LICENSE` file before open-source distribution.
