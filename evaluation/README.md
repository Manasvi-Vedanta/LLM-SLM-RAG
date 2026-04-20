# Evaluation harnesses

Runnable implementations of the experiments defined in [`../EVALUATION.md`](../EVALUATION.md).

## Status matrix (what you can run today)

| Tier | Script | Experiments | Ready? | Requires |
|------|--------|-------------|--------|----------|
| 1 | `../evaluate.py` | descriptive metrics, latency, raw agreement | ✅ | Ollama + Gemini |
| 1 | `tier1_critic_benchmark.py` | **E1** (F1 + CI), **E2** (ECE/Brier/reliability), **E4** (κ + McNemar) | ✅ | Ollama + Gemini |
| 2 | `rag_self_faithfulness.py` | **E6** self-audit, **E6'** cross-audit | ✅ | Ollama (+ Gemini for cross-audit) |
| 2 | retrieval gold harness | **E5** recall@k / nDCG | ⛔ | needs labelled `(Q, gold_chunk_id)` pairs |
| **3** | `synthetic_user_sim.py` | **E8–E11** mechanics | ✅ | pure Python, no GPU/LLM |
| 4 | user-study kit | E12–E14 | ⛔ | IRB approval, N ≥ 120 |

## Quick start

```bash
# Deterministic; safe in CI
python evaluation/synthetic_user_sim.py -v

# Requires Ollama running with gemma3-critic-v3-new
python evaluation/tier1_critic_benchmark.py --backends gemma

# Requires Ollama + GEMINI_API_KEY in .env
python evaluation/tier1_critic_benchmark.py --backends gemma gemini

# Sentence-level faithfulness audit on 10 questions
python evaluation/rag_self_faithfulness.py --questions 10
```

All scripts write JSON reports to `evaluation/results/` (gitignored).

## Adding the Qwen head-to-head

1. Fine-tune Qwen on `training_data_v3.jsonl` using the same LoRA
   config as Gemma (`finetune.py`). Publish it to Ollama as
   `qwen-critic-v1`.
2. Register it in `critic.py`: add a `QwenCritic(GemmaCritic)` subclass
   or parameterise `create_critic("qwen")` to use `OLLAMA_QWEN_MODEL`.
3. Run:

   ```bash
   python evaluation/tier1_critic_benchmark.py --backends gemma qwen gemini
   ```

4. `tier1_report.json` gets a third backend column and two extra
   pairwise agreement blocks. Paste into the paper as Table 2.

No changes to `tier1_critic_benchmark.py` are needed — it already
iterates over whatever backends you pass.
