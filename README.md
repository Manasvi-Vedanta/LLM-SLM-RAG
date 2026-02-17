# Self-Correcting RAG System (Actor-Critic)

A production-style Retrieval-Augmented Generation (RAG) project that answers questions from your PDF documents, then self-validates the retrieved context using a Critic LLM.

If retrieval is weak, the system returns **Out of Scope**.
If retrieval is good but confidence is low, the Critic generates a fallback answer from general knowledge.

---

## What this project does

This system combines:

- **Actor (Retriever):** FAISS + local sentence-transformer embeddings
- **Critic (Validator):** Gemini model that scores whether the retrieved excerpt actually answers the question
- **Gating logic:** Similarity and confidence thresholds to make behavior explicit and controllable

Core behavior:

1. Retrieve top-k chunks from indexed PDFs.
2. If best similarity is below threshold → return **Out of Scope**.
3. Else ask Critic to validate relevance and sufficiency.
4. If confidence is high → return exact document excerpt.
5. If confidence is low → return fallback answer from Critic general knowledge.

---

## Architecture

### 1) Ingestion (`ingestion.py`)

- Discovers PDF files inside `Dataset/`
- Loads pages using `PyPDFLoader`
- Splits pages into overlapping chunks using `RecursiveCharacterTextSplitter`
- Preserves metadata (`source`, `page`)

### 2) Vector Store / Actor (`vector_store.py`)

- Embeds chunks with `BAAI/bge-base-en-v1.5` (768 dimensions)
- Uses normalized embeddings and FAISS local persistence (`vectorstore/`)
- Applies query instruction prefix for BGE models:
  - `Represent this sentence for searching relevant passages:`
- Retrieves top-k and converts L2 distance to cosine similarity:

\[
\text{cosine\_sim} = 1 - \frac{L2^2}{2}
\]

### 3) Critic (`critic.py`)

- `GeminiCritic` validates whether excerpt answers question
- Returns structured confidence + explanation
- Includes retry/backoff for Gemini rate-limit (`429 ResourceExhausted`)
- `MockCritic` available for offline/testing mode

### 4) Pipeline Orchestrator (`pipeline.py`)

Implements Actor-Critic loop with two gates:

- **Scope Gate:** similarity threshold
- **Confidence Gate:** critic confidence threshold

Outputs one of:

- `document`
- `general_knowledge`
- `out_of_scope`

### 5) CLI Entry (`main.py`)

- Interactive question-answer loop
- Auto-builds vector index on first run
- Supports force rebuild and mock mode
- Handles Windows console Unicode safely (UTF-8 wrapping)

---

## Project structure

```text
Code/
├─ Dataset/
│  ├─ Principles-of-Data-Science-WEB.pdf
│  └─ Introduction_to_Python_Programming_-_WEB.pdf
├─ vectorstore/                  # generated FAISS index (ignored in git)
├─ config.py
├─ ingestion.py
├─ vector_store.py
├─ critic.py
├─ pipeline.py
├─ main.py
├─ requirements.txt
├─ .env                          # local secrets (ignored in git)
└─ README.md
```

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- Internet access (only for Gemini critic calls)

Python dependencies are in `requirements.txt`.

---

## Setup

### 1) Clone repository

```bash
git clone https://github.com/Manasvi-Vedanta/LLM-SLM-RAG.git
cd LLM-SLM-RAG
```

### 2) Create virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Add Gemini API key

Create `.env` in project root:

```env
GEMINI_API_KEY=your_real_api_key_here
```

---

## Configuration

All key knobs are in `config.py`:

- `CHUNK_SIZE = 1000`
- `CHUNK_OVERLAP = 200`
- `EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"`
- `TOP_K = 5`
- `SIMILARITY_THRESHOLD = 0.20`
- `CONFIDENCE_THRESHOLD = 85`
- `GEMINI_MODEL_NAME = "gemini-2.5-flash"`

Tuning tips:

- Increase `TOP_K` if answers seem incomplete.
- Lower `SIMILARITY_THRESHOLD` if many relevant questions become out-of-scope.
- Increase `CONFIDENCE_THRESHOLD` for stricter document-only behavior.

---

## Run

### Default interactive mode

```bash
python main.py
```

### Force rebuild index

```bash
python main.py --rebuild
```

### Offline/mock critic mode

```bash
python main.py --mock
```

### Override thresholds from CLI

```bash
python main.py --threshold 0.25 --confidence 90
```

---

## How answers are decided

For each question:

1. Actor retrieves chunk candidates.
2. If best similarity < threshold:
   - returns: `Out of Scope. The provided documents do not contain information relevant to your question.`
3. If in scope:
   - Critic validates excerpt.
4. If confidence ≥ confidence threshold:
   - returns exact document excerpt and metadata.
5. Else:
   - returns general-knowledge fallback answer.

---

## Example output modes

- **[ANSWER FROM DOCUMENT]**
  - shows source file, page, similarity, confidence
- **[ANSWER FROM GENERAL KNOWLEDGE]**
  - when document is insufficient
- **[OUT OF SCOPE]**
  - when retrieval similarity is too low

---

## Troubleshooting

### 1) `No PDF files found`

Place `.pdf` files inside `Dataset/`.

### 2) Gemini rate limit (429 / ResourceExhausted)

Handled automatically with retry/backoff in `GeminiCritic`.
If limits persist, wait and retry.

### 3) Windows Unicode / encoding crashes

`main.py` wraps stdout/stderr in UTF-8 to prevent common `cp1252` errors.

### 4) Wrong or weak retrieval

- Rebuild index: `python main.py --rebuild`
- Verify embedding model in `config.py`
- Tune `SIMILARITY_THRESHOLD`
- Ensure source PDFs have relevant content

---

## Security and repository hygiene

- `.env` is gitignored and should never be committed.
- `vectorstore/` is generated and gitignored.
- Keep API keys local only.

---

## Future improvements

- Add web UI (FastAPI/Streamlit)
- Add citation highlighting at chunk level
- Add evaluation suite and benchmark queries
- Add support for DOCX/HTML/Markdown ingestion

---

## License

No license file is included yet.
Add a `LICENSE` file before open-source distribution.
