"""
evaluate.py  –  LLM vs SLM Comparison Benchmark
-------------------------------------------------
Runs the same set of test questions through multiple Critic backends
and produces a side-by-side comparison report.

Metrics
~~~~~~~
* **JSON validity rate** – can the raw output be parsed as valid JSON?
* **Confidence calibration** – average confidence for in-scope vs out-of-scope.
* **Relevance accuracy** – does the critic correctly flag out-of-scope questions?
* **Latency** – average response time per call.
* **Agreement** – how often do both critics give the same pass/fail decision?

Usage
~~~~~
    python evaluate.py                             # compare all available backends
    python evaluate.py --backends gemini gemma      # specific pair
    python evaluate.py --out results.json           # save raw results to file
    python evaluate.py --questions 20               # limit number of questions

Re-running  after re-training the SLM lets you track improvement over time.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import config
from vector_store import load_vectorstore, retrieve
from critic import (
    BaseCritic,
    GeminiCritic,
    GemmaCritic,
    MockCritic,
    CriticResult,
    create_critic,
    parse_critic_json,
)

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
# Test question bank  (with expected scope labels)
# =====================================================================

@dataclass
class TestQuestion:
    question: str
    expected_scope: str   # "in_scope" or "out_of_scope"
    category: str         # e.g. "python", "data_science", "adversarial"


TEST_QUESTIONS: List[TestQuestion] = [
    # ── In-scope: Python ──
    TestQuestion("What is a variable in Python?", "in_scope", "python"),
    TestQuestion("How do you define a function in Python?", "in_scope", "python"),
    TestQuestion("Explain the difference between a list and a tuple.", "in_scope", "python"),
    TestQuestion("What is a dictionary in Python?", "in_scope", "python"),
    TestQuestion("How does a for loop work?", "in_scope", "python"),
    TestQuestion("What are Python's data types?", "in_scope", "python"),
    TestQuestion("What is exception handling in Python?", "in_scope", "python"),
    TestQuestion("Explain object-oriented programming.", "in_scope", "python"),
    TestQuestion("What is a lambda function?", "in_scope", "python"),
    TestQuestion("How do list comprehensions work?", "in_scope", "python"),

    # ── In-scope: Data Science ──
    TestQuestion("What is data science?", "in_scope", "data_science"),
    TestQuestion("Explain the data science lifecycle.", "in_scope", "data_science"),
    TestQuestion("What is data normalization?", "in_scope", "data_science"),
    TestQuestion("Explain standard deviation.", "in_scope", "data_science"),
    TestQuestion("What is regression analysis?", "in_scope", "data_science"),
    TestQuestion("What is supervised learning?", "in_scope", "data_science"),
    TestQuestion("Explain overfitting.", "in_scope", "data_science"),
    TestQuestion("What is a confusion matrix?", "in_scope", "data_science"),
    TestQuestion("What is cross-validation?", "in_scope", "data_science"),
    TestQuestion("Explain principal component analysis.", "in_scope", "data_science"),

    # ── Out-of-scope (adversarial) ──
    TestQuestion("What is the capital of France?", "out_of_scope", "adversarial"),
    TestQuestion("How do you cook pasta carbonara?", "out_of_scope", "adversarial"),
    TestQuestion("What is quantum computing?", "out_of_scope", "adversarial"),
    TestQuestion("Explain the rules of cricket.", "out_of_scope", "adversarial"),
    TestQuestion("What is blockchain technology?", "out_of_scope", "adversarial"),
]


# =====================================================================
# Single-question evaluation result
# =====================================================================

@dataclass
class EvalResult:
    question: str
    expected_scope: str
    category: str
    backend: str
    confidence: float
    explanation: str
    json_valid: bool
    latency_seconds: float
    similarity_score: float
    decision: str            # "document" | "general_knowledge" | "out_of_scope"
    error: Optional[str] = None


# =====================================================================
# Evaluation logic
# =====================================================================

def evaluate_question(
    critic: BaseCritic,
    backend_name: str,
    tq: TestQuestion,
    excerpt: str,
    similarity: float,
) -> EvalResult:
    """Run one question through a critic and capture the result."""
    json_valid = True
    error = None

    start = time.time()
    try:
        result: CriticResult = critic.validate(tq.question, excerpt)
    except Exception as exc:
        elapsed = time.time() - start
        return EvalResult(
            question=tq.question,
            expected_scope=tq.expected_scope,
            category=tq.category,
            backend=backend_name,
            confidence=0.0,
            explanation="",
            json_valid=False,
            latency_seconds=elapsed,
            similarity_score=similarity,
            decision="error",
            error=str(exc),
        )
    elapsed = time.time() - start

    # Decide what the pipeline would have done
    if similarity < config.SIMILARITY_THRESHOLD:
        decision = "out_of_scope"
    elif result.confidence >= config.CONFIDENCE_THRESHOLD:
        decision = "document"
    else:
        decision = "general_knowledge"

    return EvalResult(
        question=tq.question,
        expected_scope=tq.expected_scope,
        category=tq.category,
        backend=backend_name,
        confidence=result.confidence,
        explanation=result.explanation,
        json_valid=json_valid,
        latency_seconds=elapsed,
        similarity_score=similarity,
        decision=decision,
        error=error,
    )


def run_evaluation(
    backends: Dict[str, BaseCritic],
    questions: List[TestQuestion],
) -> List[EvalResult]:
    """Evaluate all questions against all backends."""
    # Load vector store
    vs = load_vectorstore()
    all_results: List[EvalResult] = []

    for i, tq in enumerate(questions, 1):
        # Retrieve chunks once (shared across backends)
        retrieval = retrieve(vs, tq.question, k=config.TOP_K)
        similarity = retrieval.best_score if retrieval.best_score else 0.0
        excerpt = (
            retrieval.best_chunk.page_content
            if retrieval.best_chunk else ""
        )

        for backend_name, critic in backends.items():
            logger.info(
                "[%d/%d] %s → %s",
                i, len(questions), backend_name, tq.question[:50],
            )
            result = evaluate_question(
                critic, backend_name, tq, excerpt, similarity
            )
            all_results.append(result)

            # Small delay between API calls
            if backend_name == "gemini":
                time.sleep(1.5)
            else:
                time.sleep(0.5)

    return all_results


# =====================================================================
# Report generation
# =====================================================================

def compute_metrics(results: List[EvalResult], backend: str) -> Dict[str, Any]:
    """Compute aggregate metrics for one backend."""
    br = [r for r in results if r.backend == backend]
    if not br:
        return {}

    in_scope = [r for r in br if r.expected_scope == "in_scope"]
    out_scope = [r for r in br if r.expected_scope == "out_of_scope"]

    # JSON validity
    json_valid_rate = sum(1 for r in br if r.json_valid) / len(br) * 100

    # Average confidence
    avg_conf_all = statistics.mean([r.confidence for r in br])
    avg_conf_in = statistics.mean([r.confidence for r in in_scope]) if in_scope else 0
    avg_conf_out = statistics.mean([r.confidence for r in out_scope]) if out_scope else 0

    # Scope accuracy: in-scope Qs should NOT be "out_of_scope", out-of-scope Qs SHOULD be
    in_scope_correct = sum(
        1 for r in in_scope if r.decision != "out_of_scope"
    )
    out_scope_correct = sum(
        1 for r in out_scope if r.decision == "out_of_scope"
    )
    scope_accuracy = (
        (in_scope_correct + out_scope_correct) / len(br) * 100
        if br else 0
    )

    # Document-answer rate for in-scope
    doc_rate = (
        sum(1 for r in in_scope if r.decision == "document") / len(in_scope) * 100
        if in_scope else 0
    )

    # Latency
    avg_latency = statistics.mean([r.latency_seconds for r in br])
    p50_latency = statistics.median([r.latency_seconds for r in br])
    max_latency = max(r.latency_seconds for r in br)

    # Error rate
    error_rate = sum(1 for r in br if r.error) / len(br) * 100

    return {
        "backend": backend,
        "total_questions": len(br),
        "json_valid_rate": round(json_valid_rate, 1),
        "avg_confidence_all": round(avg_conf_all, 1),
        "avg_confidence_in_scope": round(avg_conf_in, 1),
        "avg_confidence_out_scope": round(avg_conf_out, 1),
        "scope_accuracy": round(scope_accuracy, 1),
        "document_answer_rate": round(doc_rate, 1),
        "avg_latency_s": round(avg_latency, 2),
        "p50_latency_s": round(p50_latency, 2),
        "max_latency_s": round(max_latency, 2),
        "error_rate": round(error_rate, 1),
    }


def compute_agreement(results: List[EvalResult], backends: List[str]) -> Dict[str, Any]:
    """Compute how often two backends agree on the pass/fail decision."""
    if len(backends) < 2:
        return {}

    b1, b2 = backends[0], backends[1]
    r1_map = {r.question: r for r in results if r.backend == b1}
    r2_map = {r.question: r for r in results if r.backend == b2}

    common = set(r1_map.keys()) & set(r2_map.keys())
    if not common:
        return {}

    agree = sum(1 for q in common if r1_map[q].decision == r2_map[q].decision)
    conf_diffs = [
        abs(r1_map[q].confidence - r2_map[q].confidence)
        for q in common
    ]

    return {
        "pair": f"{b1} vs {b2}",
        "total_compared": len(common),
        "agreement_rate": round(agree / len(common) * 100, 1),
        "avg_confidence_diff": round(statistics.mean(conf_diffs), 1),
        "max_confidence_diff": round(max(conf_diffs), 1),
    }


def print_report(
    results: List[EvalResult],
    backend_names: List[str],
) -> None:
    """Pretty-print a comparison report to stdout."""
    print("\n" + "=" * 70)
    print("  EVALUATION REPORT — LLM vs SLM Comparison")
    print("=" * 70)

    # Per-backend metrics
    for bn in backend_names:
        m = compute_metrics(results, bn)
        if not m:
            continue
        print(f"\n{'─' * 40}")
        print(f"  Backend: {bn.upper()}")
        print(f"{'─' * 40}")
        print(f"  Questions tested:      {m['total_questions']}")
        print(f"  JSON validity rate:    {m['json_valid_rate']}%")
        print(f"  Avg confidence (all):  {m['avg_confidence_all']}")
        print(f"  Avg confidence (in):   {m['avg_confidence_in_scope']}")
        print(f"  Avg confidence (out):  {m['avg_confidence_out_scope']}")
        print(f"  Scope accuracy:        {m['scope_accuracy']}%")
        print(f"  Document answer rate:  {m['document_answer_rate']}%")
        print(f"  Avg latency:           {m['avg_latency_s']}s")
        print(f"  P50 latency:           {m['p50_latency_s']}s")
        print(f"  Max latency:           {m['max_latency_s']}s")
        print(f"  Error rate:            {m['error_rate']}%")

    # Agreement (if 2+ backends)
    if len(backend_names) >= 2:
        agr = compute_agreement(results, backend_names)
        if agr:
            print(f"\n{'─' * 40}")
            print(f"  Agreement: {agr['pair']}")
            print(f"{'─' * 40}")
            print(f"  Decision agreement:    {agr['agreement_rate']}%")
            print(f"  Avg confidence diff:   {agr['avg_confidence_diff']}")
            print(f"  Max confidence diff:   {agr['max_confidence_diff']}")

    # Per-question detail table
    print(f"\n{'─' * 70}")
    print("  Per-Question Results")
    print(f"{'─' * 70}")
    questions_seen = []
    for r in results:
        if r.question not in questions_seen:
            questions_seen.append(r.question)

    for q in questions_seen:
        qr = [r for r in results if r.question == q]
        print(f"\n  Q: {q}")
        for r in qr:
            marker = "OK" if r.decision != "error" else "ERR"
            print(
                f"    [{r.backend:>12}]  conf={r.confidence:5.1f}  "
                f"decision={r.decision:<18}  "
                f"latency={r.latency_seconds:.2f}s  [{marker}]"
            )

    print("\n" + "=" * 70)


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare LLM vs SLM critic backends."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to compare (default: all available). Options: gemini, gemma, mock.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Max number of test questions to use (default: all).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Save raw results as JSON to this file.",
    )
    args = parser.parse_args()

    # ── Determine which backends to test ──
    backend_names: List[str] = []
    backends: Dict[str, BaseCritic] = {}

    if args.backends:
        requested = [b.lower().strip() for b in args.backends]
    else:
        # Auto-detect available backends
        requested = []
        if config.GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            requested.append("gemini")
        # Try Ollama — check if reachable
        try:
            import requests
            resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                requested.append("gemma")
        except Exception:
            logger.info("Ollama not reachable — skipping gemma backend.")

        if not requested:
            logger.warning("No backends available. Using mock.")
            requested = ["mock"]

    for name in requested:
        try:
            critic = create_critic(name)
            backends[name] = critic
            backend_names.append(name)
            logger.info("Loaded backend: %s", name)
        except Exception as exc:
            logger.warning("Could not load backend '%s': %s", name, exc)

    if not backends:
        logger.error("No backends could be initialised. Exiting.")
        sys.exit(1)

    # ── Select test questions ──
    questions = TEST_QUESTIONS[:]
    if args.questions:
        questions = questions[: args.questions]
    logger.info("Testing %d questions against %d backends.", len(questions), len(backends))

    # ── Run evaluation ──
    results = run_evaluation(backends, questions)

    # ── Print report ──
    print_report(results, backend_names)

    # ── Save raw results ──
    if args.out:
        out_path = Path(args.out)
        raw = [asdict(r) for r in results]
        out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRaw results saved to {out_path}")

    # ── Save summary metrics ──
    summary = {
        name: compute_metrics(results, name)
        for name in backend_names
    }
    if len(backend_names) >= 2:
        summary["agreement"] = compute_agreement(results, backend_names)

    summary_path = Path("evaluation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
