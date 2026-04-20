"""
rag_self_faithfulness.py – Tier 2 self-audit of RAG pipeline answers (E6)
-------------------------------------------------------------------------
For each test question:
  1. Run the full RAG pipeline → get (answer, cited_excerpt).
  2. Split the answer into sentences.
  3. For every sentence, re-run `critic.validate(sentence, cited_excerpt)`.
  4. Flag any sentence whose confidence < FAITHFULNESS_THRESHOLD.

This is a **self-audit**, not gold-labelled validation — the same critic
is judge and defendant. Use the results to surface *internal
inconsistency* (answer drift away from cited chunk), and report it in
the paper with that caveat. For gold-labelled faithfulness you would
swap in an external judge model in `--auditor`.

Usage
-----
    python evaluation/rag_self_faithfulness.py
    python evaluation/rag_self_faithfulness.py --auditor gemini
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from critic import create_critic  # noqa: E402
from pipeline import RAGPipeline  # noqa: E402
from vector_store import load_vectorstore  # noqa: E402
from evaluate import TEST_QUESTIONS  # noqa: E402

FAITHFULNESS_THRESHOLD = 60  # below this, sentence flagged as unsupported


@dataclass
class SentenceAudit:
    sentence: str
    confidence: float
    supported: bool


@dataclass
class AnswerAudit:
    question: str
    source: str              # document | general_knowledge | out_of_scope
    n_sentences: int
    n_supported: int
    faithfulness_rate: float
    sentences: List[SentenceAudit]


def split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter — good enough for English prose.
    # Drops empties; preserves enumerated sentences.
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 3]


def audit_answer(pipeline_critic_name: str, auditor_name: str,
                 questions) -> List[AnswerAudit]:
    vs = load_vectorstore()
    pipeline = RAGPipeline(
        vectorstore=vs,
        critic=create_critic(pipeline_critic_name),
    )
    auditor = create_critic(auditor_name)

    audits: List[AnswerAudit] = []
    for i, tq in enumerate(questions, 1):
        q = tq.question
        print(f"[{i}/{len(questions)}] {q[:60]}")
        qr = pipeline.query(q)

        # Only audit document-grounded answers — fallbacks and OOS
        # responses are not claims about the corpus.
        if qr.source != "document":
            print(f"  skipped (source={qr.source})")
            audits.append(AnswerAudit(
                question=q, source=qr.source, n_sentences=0,
                n_supported=0, faithfulness_rate=1.0, sentences=[],
            ))
            continue

        # Reconstruct the excerpt that was shown to the critic
        passing = [
            ch.page_content for ch, s in zip(qr.retrieval.chunks, qr.retrieval.scores)
            if s >= pipeline.similarity_threshold
        ]
        excerpt = "\n\n---\n\n".join(passing) or qr.retrieval.best_chunk.page_content

        sents = split_sentences(qr.answer)
        sent_audits: List[SentenceAudit] = []
        for s in sents:
            res = auditor.validate(s, excerpt)
            supported = res.confidence >= FAITHFULNESS_THRESHOLD
            sent_audits.append(SentenceAudit(
                sentence=s, confidence=res.confidence, supported=supported))
            time.sleep(0.3 if auditor_name != "gemini" else 1.0)

        supported_n = sum(1 for s in sent_audits if s.supported)
        rate = supported_n / len(sent_audits) if sent_audits else 1.0

        audits.append(AnswerAudit(
            question=q, source=qr.source, n_sentences=len(sent_audits),
            n_supported=supported_n, faithfulness_rate=round(rate, 3),
            sentences=sent_audits,
        ))
        print(f"  sentences={len(sent_audits)}  supported={supported_n}  "
              f"rate={rate:.2f}")

    return audits


def summarise(audits: List[AnswerAudit]) -> dict:
    document_audits = [a for a in audits if a.source == "document" and a.n_sentences > 0]
    if not document_audits:
        return {"n_document_answers": 0}

    rates = [a.faithfulness_rate for a in document_audits]
    return {
        "n_answers_audited": len(audits),
        "n_document_answers": len(document_audits),
        "mean_faithfulness": round(stats.mean(rates), 3),
        "median_faithfulness": round(stats.median(rates), 3),
        "min_faithfulness": round(min(rates), 3),
        "answers_below_0_7": sum(1 for r in rates if r < 0.7),
        "answers_above_0_9": sum(1 for r in rates if r >= 0.9),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", default="gemma",
                    help="Critic used inside the pipeline")
    ap.add_argument("--auditor", default="gemma",
                    help="Critic used to audit each sentence (self=gemma, external=gemini)")
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--out", default="evaluation/results/tier2_faithfulness.json")
    args = ap.parse_args()

    questions = TEST_QUESTIONS[: args.questions]
    audits = audit_answer(args.critic, args.auditor, questions)
    summary = summarise(audits)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "pipeline_critic": args.critic,
        "auditor": args.auditor,
        "self_audit": args.critic == args.auditor,
        "threshold": FAITHFULNESS_THRESHOLD,
        "summary": summary,
        "audits": [asdict(a) for a in audits],
    }, indent=2))

    print("\n" + "=" * 56)
    print(f"  Faithfulness audit (self={args.critic == args.auditor})")
    print("=" * 56)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
