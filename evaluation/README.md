# Evaluation harnesses

Runnable implementations of the experiments defined in [`../EVALUATION.md`](../EVALUATION.md).

| Tier | Script | Experiments | Requires |
|------|--------|-------------|----------|
| 1 | `../evaluate.py` | E1, E4 | Ollama (local) + Gemini key |
| 1 | `critic_calibration.py` *(stub)* | E2 | Ollama |
| 2 | `retrieval_metrics.py` *(stub)* | E5 | FAISS index + gold labels |
| 2 | `rag_faithfulness.py` *(stub)* | E6, E7 | Ollama + FAISS |
| **3** | `synthetic_user_sim.py` | **E8–E11** | pure Python, no GPU/LLM |
| 4 | User-study kit *(out of scope)* | E12–E14 | IRB approval |

Tier 3 is the only one that is fully offline and CI-safe. Run it before
every commit that touches `mastery_tracker.py`, `review_scheduler.py`,
`question_generator.py`, or `knowledge_transfer.py`:

```bash
python evaluation/synthetic_user_sim.py -v
```

Exit code is non-zero if any invariant fails — wire it into a pre-push
hook or GitHub Actions job once you're ready to gate merges on it.

Stub harnesses for Tiers 1–2 are tracked in EVALUATION.md; implement
them incrementally as the labelled datasets come online.
