"""
generate_training_data.py  –  Knowledge-Distillation Data Generator
--------------------------------------------------------------------
Uses the **Gemini critic** (strong teacher LLM) to produce labelled
training examples that can later fine-tune the **Gemma 3 4B** SLM.

Workflow
~~~~~~~~
1. Load the FAISS vector store.
2. Run a bank of seed questions (+ optional synthetic questions).
3. For each question, retrieve top-k chunks and ask Gemini to validate.
4. Log every (prompt, completion) pair into a JSONL file.

The output file (``training_data.jsonl``) is directly consumable by
``finetune.py``.

Usage
~~~~~
    python generate_training_data.py                  # default 5 seeds × top-k
    python generate_training_data.py --out data.jsonl # custom output path
    python generate_training_data.py --expand         # also generate synthetic Qs

Re-running appends to the file so you can grow the dataset over time.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import config
from ingestion import ingest
from vector_store import load_vectorstore, retrieve, build_vectorstore
from critic import GeminiCritic, CriticResult

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── encoding fix for Windows consoles ────────────────────────────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# =====================================================================
# Seed question bank
# =====================================================================
# Mix of in-scope (should match your PDFs) and out-of-scope questions.
# Add more to improve dataset diversity.  50+ unique seeds recommended.

SEED_QUESTIONS: List[str] = [
    # ── In-scope: Python programming ──
    "What is a variable in Python?",
    "How do you define a function in Python?",
    "What are the different data types in Python?",
    "Explain the difference between a list and a tuple.",
    "What is a dictionary in Python?",
    "How does a for loop work in Python?",
    "What is a while loop?",
    "Explain conditional statements in Python.",
    "What is string slicing in Python?",
    "How do you handle exceptions in Python?",
    "What are Python modules and packages?",
    "Explain object-oriented programming in Python.",
    "What is a class in Python?",
    "How does inheritance work in Python?",
    "What is a lambda function?",
    "What are list comprehensions?",
    "How do you read and write files in Python?",
    "What is the difference between == and is in Python?",
    "Explain the concept of indentation in Python.",
    "What is recursion?",

    # ── In-scope: Data Science ──
    "What is data science?",
    "Explain the data science lifecycle.",
    "What is exploratory data analysis?",
    "What is data normalization?",
    "Explain standard deviation.",
    "What is the difference between population and sample?",
    "What is a probability distribution?",
    "Explain Bayes' theorem.",
    "What is regression analysis?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of overfitting.",
    "What is a confusion matrix?",
    "What is cross-validation?",
    "Explain the bias-variance tradeoff.",
    "What is feature engineering?",
    "What is dimensionality reduction?",
    "Explain principal component analysis.",
    "What is clustering?",
    "What are decision trees?",
    "Explain the k-nearest neighbours algorithm.",

    # ── Out-of-scope (adversarial — model should reject) ──
    "What is the capital of France?",
    "Who won the 2024 Super Bowl?",
    "How do you cook pasta carbonara?",
    "What is quantum computing?",
    "Explain general relativity.",
    "What is blockchain technology?",
    "How does the stock market work?",
    "What is photosynthesis?",
    "Explain the rules of cricket.",
    "What is the speed of light?",
]

# ── Paraphrase templates for data augmentation ──
_PARAPHRASE_TEMPLATES = [
    "Can you explain {concept}?",
    "Describe {concept} in simple terms.",
    "What do you mean by {concept}?",
    "Give me a brief overview of {concept}.",
    "How would you define {concept}?",
]

_CONCEPTS_FOR_AUGMENTATION = [
    "variables in Python",
    "data types",
    "loops",
    "functions",
    "classes and objects",
    "exception handling",
    "data normalization",
    "standard deviation",
    "regression",
    "overfitting",
    "cross-validation",
    "clustering",
    "decision trees",
    "probability distributions",
    "feature engineering",
]


def expand_questions(base_questions: List[str]) -> List[str]:
    """Generate additional paraphrased questions for data augmentation."""
    extra: List[str] = []
    for concept in _CONCEPTS_FOR_AUGMENTATION:
        template = random.choice(_PARAPHRASE_TEMPLATES)
        extra.append(template.format(concept=concept))
    return base_questions + extra


# =====================================================================
# The validation prompt (identical to GeminiCritic._VALIDATE_PROMPT)
# =====================================================================
VALIDATE_PROMPT = """\
You are a strict validation judge for a Retrieval-Augmented Generation system.

### TASK
Decide whether the **Excerpt** below correctly and sufficiently answers the
**Question**.  Return your assessment as JSON with exactly two keys:

  {{"confidence": <int 0-100>, "explanation": "<one sentence>"}}

Rules:
* 100 = the excerpt fully and correctly answers the question.
*   0 = the excerpt is completely irrelevant.
* Be harsh — partial or vague matches should score below 70.

### QUESTION
{question}

### EXCERPT
{excerpt}

### YOUR JSON RESPONSE (no markdown fences):
"""


# =====================================================================
# Data generation
# =====================================================================

def generate_single_example(
    critic: GeminiCritic,
    question: str,
    excerpt: str,
    source_file: str,
    page: Any,
    similarity: float,
    rate_limit_delay: float = 2.0,
) -> Dict[str, Any] | None:
    """
    Ask Gemini to validate one (question, excerpt) pair and return
    a training record. Returns None if the API call fails.
    """
    prompt = VALIDATE_PROMPT.format(question=question, excerpt=excerpt)

    try:
        result: CriticResult = critic.validate(question, excerpt)
    except Exception as exc:
        logger.warning("Gemini call failed for '%s': %s", question[:50], exc)
        return None

    record = {
        "prompt": prompt,
        "completion": json.dumps({
            "confidence": result.confidence,
            "explanation": result.explanation,
        }),
        # ── metadata (not used in training, but useful for analysis) ──
        "question": question,
        "source_file": source_file,
        "page": page,
        "similarity": round(similarity, 4),
    }

    time.sleep(rate_limit_delay)  # respect Gemini rate limits
    return record


def generate_dataset(
    questions: List[str],
    output_path: Path,
    rate_limit_delay: float = 2.0,
) -> int:
    """
    Run every question through the vector store + Gemini critic and
    append results to *output_path* as JSONL.

    Returns the number of examples generated.
    """
    # ── Load or build the vector store ──
    if config.VECTORSTORE_DIR.exists():
        logger.info("Loading existing FAISS index …")
        vs = load_vectorstore()
    else:
        logger.info("No FAISS index found — building from PDFs …")
        chunks = ingest()
        vs = build_vectorstore(chunks)

    # ── Instantiate the Gemini teacher ──
    if config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error(
            "GEMINI_API_KEY is not set.  Training-data generation requires "
            "the Gemini API.  Please set GEMINI_API_KEY in your .env file."
        )
        return 0

    critic = GeminiCritic()
    count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for i, question in enumerate(questions, 1):
            logger.info(
                "[%d/%d] Processing: %s", i, len(questions), question[:60]
            )

            # Retrieve top-k chunks
            retrieval = retrieve(vs, question, k=config.TOP_K)

            if not retrieval.chunks:
                logger.info("  -> No chunks retrieved, skipping.")
                continue

            # For EACH retrieved chunk, generate a separate training example.
            # This multiplies the dataset by ~TOP_K for in-scope questions.
            for chunk, score in zip(retrieval.chunks, retrieval.scores):
                source_file = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", "?")

                record = generate_single_example(
                    critic=critic,
                    question=question,
                    excerpt=chunk.page_content,
                    source_file=source_file,
                    page=page,
                    similarity=score,
                    rate_limit_delay=rate_limit_delay,
                )

                if record is not None:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    count += 1
                    logger.info(
                        "  -> Saved example #%d  (sim=%.3f  conf=%s)",
                        count,
                        score,
                        json.loads(record["completion"])["confidence"],
                    )

    return count


# =====================================================================
# CLI entry point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning training data using Gemini as teacher."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="training_data.jsonl",
        help="Output JSONL file path (default: training_data.jsonl).",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Also generate paraphrased / augmented questions.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between Gemini calls (default: 2.0).",
    )
    args = parser.parse_args()

    questions = SEED_QUESTIONS[:]
    if args.expand:
        questions = expand_questions(questions)
        logger.info("Expanded question set to %d questions.", len(questions))
    else:
        logger.info("Using %d seed questions.", len(questions))

    random.shuffle(questions)

    output_path = Path(args.out)
    existing = 0
    if output_path.exists():
        existing = sum(1 for _ in open(output_path, encoding="utf-8"))
        logger.info("Appending to existing file (%d records already).", existing)

    count = generate_dataset(questions, output_path, rate_limit_delay=args.delay)

    total = existing + count
    print(f"\nDone!  Generated {count} new examples.")
    print(f"Total examples in {output_path}: {total}")
    print(f"\nNext step:  python finetune.py --data {output_path}")


if __name__ == "__main__":
    main()
