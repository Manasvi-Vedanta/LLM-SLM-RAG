"""
generate_synthetic_data.py  –  Synthetic Training Data Generator
-----------------------------------------------------------------
Generates training data for fine-tuning the Gemma critic **without**
any LLM API calls.  Uses deterministic heuristics based on:

  * Cosine similarity between query and chunk (from FAISS)
  * Whether the question is in-scope or out-of-scope
  * Keyword overlap between question and chunk content

This is combined with the 25 Gemini-labelled examples from
``training_data.jsonl`` to produce a complete training set.

Usage
~~~~~
    python generate_synthetic_data.py                     # default output
    python generate_synthetic_data.py --out synth.jsonl   # custom path
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import config
from vector_store import load_vectorstore, retrieve, _get_embeddings
from ingestion import ingest

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
# Questions with ground-truth labels
# =====================================================================

IN_SCOPE_QUESTIONS = [
    # Python programming
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
    "What are Python operators?",
    "Explain the range function in Python.",
    "What is a set in Python?",
    "How do you import modules in Python?",
    "What are default arguments in functions?",
    "What is the self keyword in Python?",
    "How do you create a list in Python?",
    "What is type casting?",
    "What are f-strings in Python?",
    "How do you handle multiple exceptions?",

    # Data Science
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
    "What is the mean of a dataset?",
    "What is a histogram?",
    "Explain correlation and causation.",
    "What is a scatter plot?",
    "What are outliers?",
]

OUT_OF_SCOPE_QUESTIONS = [
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
    "What is the tallest building in the world?",
    "How does WiFi work?",
    "What is the chemical formula for water?",
    "Who painted the Mona Lisa?",
    "What is cryptocurrency mining?",
    "How do airplanes fly?",
    "What is the population of China?",
    "Explain the water cycle.",
    "What are black holes?",
    "What is DNA replication?",
]

# ── Variations for augmentation (paraphrases) ──
PARAPHRASE_TEMPLATES = [
    "Can you explain {concept}?",
    "Describe {concept} in simple terms.",
    "What do you mean by {concept}?",
    "Give me a brief overview of {concept}.",
    "How would you define {concept}?",
    "What is the concept of {concept}?",
]

IN_SCOPE_CONCEPTS = [
    "variables in Python", "data types", "loops in Python",
    "functions in Python", "classes and objects", "exception handling",
    "data normalization", "standard deviation", "regression analysis",
    "overfitting", "cross-validation", "clustering algorithms",
    "decision trees", "probability distributions", "feature engineering",
    "list comprehensions", "inheritance in OOP", "recursion",
    "Bayes theorem", "dimensionality reduction",
]


# =====================================================================
# The validation prompt template (matches critic.py exactly)
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
# Explanation templates (realistic Critic-style explanations)
# =====================================================================

HIGH_CONF_EXPLANATIONS = [
    "The excerpt directly and thoroughly answers the question with relevant detail.",
    "The excerpt provides a clear and complete answer to the question.",
    "The passage contains a comprehensive explanation that fully addresses the question.",
    "The excerpt directly covers the topic and provides sufficient detail to answer the question.",
    "The document excerpt accurately and completely addresses the question asked.",
    "The retrieved passage fully answers the question with specific, accurate information.",
    "The excerpt contains all the information needed to answer the question correctly.",
    "The document provides a thorough and precise answer to this specific question.",
]

MEDIUM_CONF_EXPLANATIONS = [
    "The excerpt partially addresses the question but lacks full detail or specificity.",
    "The passage touches on the topic but does not fully or directly answer the question.",
    "The excerpt contains related information but only partially answers the question.",
    "Some relevant information is present but the excerpt does not comprehensively address the question.",
    "The passage is tangentially related but does not provide a complete answer.",
]

LOW_CONF_EXPLANATIONS = [
    "The excerpt does not contain information relevant to the question.",
    "The passage is unrelated to the question asked.",
    "The excerpt discusses a different topic entirely and does not address the question.",
    "There is no meaningful overlap between the question and the excerpt content.",
    "The document excerpt fails to address the question in any substantive way.",
]


# =====================================================================
# Confidence assignment heuristics
# =====================================================================

def compute_keyword_overlap(question: str, excerpt: str) -> float:
    """Fraction of question words found in the excerpt (0.0–1.0)."""
    stop_words = {
        "what", "is", "the", "a", "an", "in", "of", "and", "or", "to",
        "how", "do", "does", "you", "are", "can", "explain", "describe",
        "between", "give", "me", "brief", "overview", "define", "concept",
        "that", "this", "for", "with", "from", "by", "on", "it", "its",
        "be", "was", "were", "been", "being", "have", "has", "had",
    }
    q_words = set(question.lower().split()) - stop_words
    if not q_words:
        return 0.0
    excerpt_lower = excerpt.lower()
    matches = sum(1 for w in q_words if w in excerpt_lower)
    return matches / len(q_words)


def assign_confidence(
    similarity: float,
    keyword_overlap: float,
    is_in_scope: bool,
) -> Tuple[int, str]:
    """
    Deterministically assign a confidence score and explanation
    based on similarity, keyword overlap, and scope label.

    Calibration targets (matching config.CONFIDENCE_THRESHOLD = 85):
      - In-scope + high similarity (>=0.80) → 85-100  ("document" answer)
      - In-scope + moderate similarity        → 50-84   ("general_knowledge")
      - In-scope + low similarity             → 10-49
      - Out-of-scope                          → 0-25

    This ensures ~35-40 % of in-scope examples score >= 85 so the
    fine-tuned model learns when to trust the retrieved excerpt.
    """
    if not is_in_scope:
        # Out-of-scope questions should always get low confidence
        if similarity < 0.25:
            conf = random.randint(0, 5)
        elif similarity < 0.40:
            conf = random.randint(0, 15)
        else:
            # Some out-of-scope Qs may have moderate similarity
            # (e.g. "quantum computing" vs a general science excerpt)
            conf = random.randint(5, 25)
        explanation = random.choice(LOW_CONF_EXPLANATIONS)
        return conf, explanation

    # ── In-scope questions ──
    # Weight similarity more heavily — it's the stronger signal
    combined = similarity * 0.7 + keyword_overlap * 0.3

    if combined >= 0.65:
        # High confidence band: 85-100 → triggers "document" answers
        # similarity >= ~0.80 with decent keyword overlap lands here
        base = int(85 + (combined - 0.65) * 43)   # 85 at 0.65 → 100 at 1.0
        conf = min(100, max(85, base + random.randint(-3, 3)))
        explanation = random.choice(HIGH_CONF_EXPLANATIONS)
    elif combined >= 0.50:
        # Medium-high confidence band: 60-84
        base = int(60 + (combined - 0.50) * 160)   # 60 at 0.50 → 84 at 0.65
        conf = min(84, max(60, base + random.randint(-4, 4)))
        explanation = random.choice(MEDIUM_CONF_EXPLANATIONS)
    elif combined >= 0.35:
        # Medium-low band: 35-59
        base = int(35 + (combined - 0.35) * 160)
        conf = min(59, max(35, base + random.randint(-4, 4)))
        explanation = random.choice(MEDIUM_CONF_EXPLANATIONS)
    elif combined >= 0.20:
        # Low band: 10-34
        base = int(10 + (combined - 0.20) * 160)
        conf = min(34, max(10, base + random.randint(-3, 3)))
        explanation = random.choice(LOW_CONF_EXPLANATIONS)
    else:
        # Very low: 0-15
        conf = random.randint(0, 15)
        explanation = random.choice(LOW_CONF_EXPLANATIONS)

    return conf, explanation


# =====================================================================
# Data generation
# =====================================================================

def generate_examples_for_question(
    vs,
    question: str,
    is_in_scope: bool,
    max_chunks: int = 3,
) -> List[Dict[str, Any]]:
    """Generate training examples for a single question."""
    retrieval = retrieve(vs, question, k=config.TOP_K)

    if not retrieval.chunks:
        return []

    examples = []
    for chunk, score in zip(
        retrieval.chunks[:max_chunks],
        retrieval.scores[:max_chunks],
    ):
        excerpt = chunk.page_content
        source_file = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")

        keyword_overlap = compute_keyword_overlap(question, excerpt)
        confidence, explanation = assign_confidence(score, keyword_overlap, is_in_scope)

        prompt = VALIDATE_PROMPT.format(question=question, excerpt=excerpt)
        completion = json.dumps({
            "confidence": confidence,
            "explanation": explanation,
        })

        examples.append({
            "prompt": prompt,
            "completion": completion,
            "question": question,
            "source_file": source_file,
            "page": page,
            "similarity": round(score, 4),
            "keyword_overlap": round(keyword_overlap, 3),
            "is_in_scope": is_in_scope,
            "label_source": "synthetic",
        })

    return examples


def generate_augmented_questions() -> List[Tuple[str, bool]]:
    """Generate paraphrased in-scope questions for augmentation."""
    augmented = []
    for concept in IN_SCOPE_CONCEPTS:
        template = random.choice(PARAPHRASE_TEMPLATES)
        augmented.append((template.format(concept=concept), True))
    return augmented


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data (no API calls needed)."
    )
    parser.add_argument(
        "--out", type=str, default="training_data.jsonl",
        help="Output JSONL file (appends to existing). Default: training_data.jsonl",
    )
    parser.add_argument(
        "--chunks-per-question", type=int, default=3,
        help="Max chunks per question (default: 3).",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Skip paraphrase augmentation.",
    )
    args = parser.parse_args()

    output_path = Path(args.out)

    # Count existing examples
    existing = 0
    if output_path.exists():
        existing = sum(1 for _ in open(output_path, encoding="utf-8"))
        logger.info("Existing examples in %s: %d", output_path, existing)

    # Load vector store
    logger.info("Loading FAISS index …")
    vs = load_vectorstore()

    # Build question list
    all_questions: List[Tuple[str, bool]] = []
    for q in IN_SCOPE_QUESTIONS:
        all_questions.append((q, True))
    for q in OUT_OF_SCOPE_QUESTIONS:
        all_questions.append((q, False))

    if not args.no_augment:
        augmented = generate_augmented_questions()
        all_questions.extend(augmented)
        logger.info("Added %d augmented questions.", len(augmented))

    random.shuffle(all_questions)
    logger.info("Total questions to process: %d", len(all_questions))

    # Generate
    count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for i, (question, is_in_scope) in enumerate(all_questions, 1):
            examples = generate_examples_for_question(
                vs, question, is_in_scope, max_chunks=args.chunks_per_question,
            )
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1

            if i % 10 == 0:
                logger.info("Processed %d/%d questions (%d examples so far)", i, len(all_questions), count)

    total = existing + count
    print(f"\nDone!  Generated {count} synthetic examples.")
    print(f"Total examples in {output_path}: {total}")
    print(f"  - Gemini-labelled: {existing}")
    print(f"  - Synthetic: {count}")
    print(f"\nNext step:  python finetune.py --data {output_path}")


if __name__ == "__main__":
    main()
