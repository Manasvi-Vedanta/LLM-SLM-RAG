"""
tier1_critic_benchmark.py – Tier 1 critic evaluation (E1, E2, E4)
------------------------------------------------------------------
Extends `evaluate.py` with publication-grade metrics that the original
side-by-side benchmark did not produce:

- E1  Binary classification: accuracy, precision, recall, F1, confusion
      matrix for "in-scope pass" decision. McNemar's statistic between
      backends on the same items.
- E2  Calibration: Expected Calibration Error (ECE, 10-bin), Brier score,
      and a reliability table.
- E4  Cross-backend agreement: raw agreement %, Cohen's κ.

Uses the **existing 25-question test bank** in `evaluate.py` (20 in-scope,
5 adversarial out-of-scope), so this runs today against the currently
deployed `gemma3-critic-v3-new` + Gemini cloud fallback. When the Qwen
fine-tune is ready, add it to `--backends` and the same metrics populate
automatically.

Usage
-----
    python evaluation/tier1_critic_benchmark.py --backends gemma gemini
    python evaluation/tier1_critic_benchmark.py --backends gemma
    python evaluation/tier1_critic_benchmark.py --out tier1_report.json

Important caveats (document in the paper):
* n = 25 is small. Metrics are reported with bootstrap 95% CI so reviewers
  see the width of the uncertainty.
* The 20 in-scope questions were hand-crafted for evaluate.py and have
  overlap with the training distribution. To claim *held-out* accuracy
  you must replace them with a labelled held-out file (see EVALUATION.md
  §Datasets).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as stats
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from critic import create_critic, BaseCritic, CriticResult  # noqa: E402
from vector_store import load_vectorstore, retrieve  # noqa: E402

# Pull the existing question bank so we don't duplicate it.
from evaluate import TEST_QUESTIONS, TestQuestion  # noqa: E402


# =====================================================================
# Per-item record
# =====================================================================
@dataclass
class Record:
    question: str
    expected_scope: str       # "in_scope" | "out_of_scope"
    category: str
    backend: str
    confidence: float         # 0–100
    prediction_pass: bool     # True if system would surface as document-grounded
    similarity: float
    latency_s: float
    error: str | None = None


def collect_backend(
    critic: BaseCritic, name: str, questions: List[TestQuestion]
) -> List[Record]:
    vs = load_vectorstore()
    out: List[Record] = []
    for i, tq in enumerate(questions, 1):
        retrieval = retrieve(vs, tq.question, k=config.TOP_K)
        sim = retrieval.best_score or 0.0
        excerpt = retrieval.best_chunk.page_content if retrieval.best_chunk else ""

        start = time.time()
        try:
            res: CriticResult = critic.validate(tq.question, excerpt)
            err = None
            conf = res.confidence
        except Exception as exc:
            res = None
            err = str(exc)
            conf = 0.0
        elapsed = time.time() - start

        # Pipeline decision: in-scope document-grounded iff both gates pass.
        passes = (
            sim >= config.SIMILARITY_THRESHOLD
            and conf >= config.CONFIDENCE_THRESHOLD
            and err is None
        )
        out.append(Record(
            question=tq.question,
            expected_scope=tq.expected_scope,
            category=tq.category,
            backend=name,
            confidence=conf,
            prediction_pass=passes,
            similarity=sim,
            latency_s=elapsed,
            error=err,
        ))
        print(f"  [{i:2d}/{len(questions)}] {name:>8s}  "
              f"conf={conf:5.1f}  pass={passes}  "
              f"({tq.expected_scope})  {elapsed:.2f}s")
        time.sleep(1.2 if name == "gemini" else 0.3)
    return out


# =====================================================================
# E1 — binary classification metrics
# =====================================================================
def _confusion(records: List[Record]) -> dict:
    tp = fp = tn = fn = 0
    for r in records:
        y = 1 if r.expected_scope == "in_scope" else 0
        p = 1 if r.prediction_pass else 0
        if y == 1 and p == 1: tp += 1
        elif y == 0 and p == 0: tn += 1
        elif y == 0 and p == 1: fp += 1
        else: fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _f1_from_conf(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
    }


def _bootstrap_ci(records: List[Record], metric_fn, n_boot: int = 2000,
                  seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    samples = []
    n = len(records)
    if n == 0:
        return (0.0, 0.0)
    for _ in range(n_boot):
        resample = [records[rng.randrange(n)] for _ in range(n)]
        samples.append(metric_fn(resample))
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot)]
    return (round(lo, 4), round(hi, 4))


def e1_metrics(records: List[Record]) -> dict:
    cm = _confusion(records)
    m = _f1_from_conf(cm)
    f1_ci = _bootstrap_ci(records, lambda rs: _f1_from_conf(_confusion(rs))["f1"])
    acc_ci = _bootstrap_ci(records, lambda rs: _f1_from_conf(_confusion(rs))["accuracy"])
    return {**m, "confusion": cm, "f1_95ci": f1_ci, "accuracy_95ci": acc_ci}


# =====================================================================
# E2 — calibration (ECE, Brier, reliability table)
# =====================================================================
def e2_calibration(records: List[Record], n_bins: int = 10) -> dict:
    """Treat confidence/100 as predicted probability that the item is in-scope.
    Ground truth = 1 if in_scope else 0. Compute binned ECE + Brier."""
    probs = [r.confidence / 100.0 for r in records]
    labels = [1 if r.expected_scope == "in_scope" else 0 for r in records]

    # Brier score (lower is better; perfectly calibrated → 0)
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(records)

    # ECE
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))

    ece = 0.0
    reliability: list[dict] = []
    for i, bucket in enumerate(bins):
        if not bucket:
            reliability.append({"bin": i, "count": 0, "avg_conf": None,
                               "accuracy": None})
            continue
        avg_p = stats.mean(p for p, _ in bucket)
        acc   = stats.mean(y for _, y in bucket)
        ece  += (len(bucket) / len(records)) * abs(avg_p - acc)
        reliability.append({
            "bin": i,
            "count": len(bucket),
            "avg_conf": round(avg_p, 3),
            "accuracy": round(acc, 3),
            "gap": round(avg_p - acc, 3),
        })

    return {
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "reliability": reliability,
    }


# =====================================================================
# E4 — agreement & Cohen's κ
# =====================================================================
def _cohen_kappa(pairs: List[Tuple[int, int]]) -> float:
    """pairs = [(pred_A, pred_B), ...] with values in {0,1}."""
    n = len(pairs)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in pairs if a == b) / n
    pA1 = sum(a for a, _ in pairs) / n
    pB1 = sum(b for _, b in pairs) / n
    chance = pA1 * pB1 + (1 - pA1) * (1 - pB1)
    if chance >= 1.0:
        return 1.0
    return round((agree - chance) / (1 - chance), 4)


def _mcnemar_stat(pairs: List[Tuple[int, int, int]]) -> dict:
    """pairs = [(y_true, pred_A, pred_B)]. Compute McNemar's 2x2 on disagreements."""
    b = sum(1 for y, a, c in pairs if a == y and c != y)  # A right, B wrong
    c = sum(1 for y, a, cc in pairs if a != y and cc == y)  # B right, A wrong
    # Continuity-corrected statistic
    if b + c == 0:
        stat = 0.0
    else:
        stat = ((abs(b - c) - 1) ** 2) / (b + c)
    return {"b_A_right_B_wrong": b, "c_A_wrong_B_right": c, "chi2_stat": round(stat, 3)}


def e4_agreement(by_backend: dict) -> dict:
    names = list(by_backend.keys())
    if len(names) < 2:
        return {}
    A, B = names[0], names[1]
    a_map = {r.question: r for r in by_backend[A]}
    b_map = {r.question: r for r in by_backend[B]}
    common = [q for q in a_map if q in b_map]

    pairs = [(int(a_map[q].prediction_pass), int(b_map[q].prediction_pass))
             for q in common]
    raw_agree = sum(1 for a, b in pairs if a == b) / len(pairs) * 100 if pairs else 0.0
    kappa = _cohen_kappa(pairs)

    # McNemar needs ground truth
    labeled = [(1 if a_map[q].expected_scope == "in_scope" else 0,
                int(a_map[q].prediction_pass),
                int(b_map[q].prediction_pass))
               for q in common]
    mc = _mcnemar_stat(labeled)

    return {
        "pair": f"{A} vs {B}",
        "n": len(common),
        "raw_agreement_pct": round(raw_agree, 1),
        "cohens_kappa": kappa,
        "mcnemar": mc,
    }


# =====================================================================
# Latency summary
# =====================================================================
def latency_summary(records: List[Record]) -> dict:
    lats = [r.latency_s for r in records if r.error is None]
    if not lats:
        return {"p50": None, "p95": None, "mean": None}
    lats_sorted = sorted(lats)
    p50 = lats_sorted[len(lats_sorted) // 2]
    p95 = lats_sorted[int(0.95 * len(lats_sorted))]
    return {
        "mean_s": round(stats.mean(lats), 3),
        "p50_s": round(p50, 3),
        "p95_s": round(p95, 3),
        "max_s": round(max(lats), 3),
    }


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["gemma"],
                    help="Subset of: gemma gemini mock")
    ap.add_argument("--questions", type=int, default=None,
                    help="Cap test-set size (default: all 25)")
    ap.add_argument("--out", default="evaluation/results/tier1_report.json")
    args = ap.parse_args()

    questions = TEST_QUESTIONS[: args.questions] if args.questions else TEST_QUESTIONS
    print(f"Testing {len(questions)} questions against {args.backends}\n")

    by_backend: dict[str, List[Record]] = {}
    for name in args.backends:
        try:
            critic = create_critic(name)
        except Exception as exc:
            print(f"Could not load backend {name}: {exc}")
            continue
        print(f"── Backend: {name} ──")
        by_backend[name] = collect_backend(critic, name, questions)
        print()

    report = {"n_questions": len(questions), "backends": {}}
    for name, recs in by_backend.items():
        report["backends"][name] = {
            "E1_classification": e1_metrics(recs),
            "E2_calibration":    e2_calibration(recs),
            "latency":           latency_summary(recs),
            "error_rate":        round(sum(1 for r in recs if r.error) / len(recs), 4),
        }
    if len(by_backend) >= 2:
        report["E4_agreement"] = e4_agreement(by_backend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    # Human-readable summary
    print("\n" + "=" * 64)
    print("  Tier-1 Publication-Grade Metrics")
    print("=" * 64)
    for name, b in report["backends"].items():
        print(f"\n  [{name}]")
        print(f"    Accuracy         : {b['E1_classification']['accuracy']}  "
              f"(95% CI {b['E1_classification']['accuracy_95ci']})")
        print(f"    F1               : {b['E1_classification']['f1']}  "
              f"(95% CI {b['E1_classification']['f1_95ci']})")
        print(f"    Precision/Recall : {b['E1_classification']['precision']} / "
              f"{b['E1_classification']['recall']}")
        print(f"    ECE              : {b['E2_calibration']['ece']}")
        print(f"    Brier            : {b['E2_calibration']['brier']}")
        print(f"    Latency p50 / p95: {b['latency']['p50_s']}s / "
              f"{b['latency']['p95_s']}s")
    if "E4_agreement" in report:
        a = report["E4_agreement"]
        print(f"\n  [{a['pair']}]  κ={a['cohens_kappa']}  "
              f"raw={a['raw_agreement_pct']}%  "
              f"McNemar χ²={a['mcnemar']['chi2_stat']}")
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
