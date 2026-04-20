# Evaluation Protocol

This document defines the experimental protocol used to evaluate the
Self-Correcting RAG Adaptive Learning Assistant. Every claim in the README
— "the critic is calibrated", "decay drives spaced review correctly",
"ZPD targeting lifts mastery faster than uniform difficulty" — must be
backed by one of the experiments below before the paper goes out.

The protocol is organised in **four tiers** corresponding to four units
of analysis:

| Tier | Unit | Experiments | Runnable harness |
|------|------|-------------|------------------|
| T1 | **Critic model** (fine-tuned SLM) | E1–E4 | `evaluate.py`, `evaluation/tier1_critic_benchmark.py` |
| T2 | **RAG pipeline** (Actor + Critic + gates) | E5–E7 | `evaluation/rag_self_faithfulness.py` |
| T3 | **Adaptive analytics modules** (offline, synthetic) | E8–E11 | `evaluation/synthetic_user_sim.py` |
| T4 | **End-to-end human learning outcomes** | E12–E14 | IRB-gated user study |

Each experiment declares: **RQ** (research question), **H** (hypothesis),
**metric**, **design**, **statistics**, and **success criterion**.

### What you can run **today** (no extra labelling, no IRB)

| Experiment | Runnable now? | Command | Bottleneck |
|------------|---------------|---------|------------|
| **E1** accuracy / F1 / precision / recall | ✅ yes (n = 25) | `python evaluation/tier1_critic_benchmark.py --backends gemma gemini` | small n — use bootstrap CI, paper must state "pilot sample" until held-out set is labelled |
| **E2** ECE / Brier / reliability table | ✅ yes (n = 25) | same as above | same |
| **E3** adversarial scope robustness | ⚠️ partial | same as above (only 5 OOS items) | need 100 near-topic distractors |
| **E4** κ + McNemar agreement | ✅ yes, with ≥ 2 backends | same as above | requires Gemini key + Ollama up |
| **E4b** Qwen head-to-head | 🕒 waiting on Qwen LoRA | `tier1_critic_benchmark.py --backends gemma qwen gemini` after fine-tune | must train Qwen on `training_data_v3.jsonl` first |
| **E5** retrieval recall@k, nDCG | ❌ blocked | needs labelled `(question, gold_chunk_id)` pairs | label 150 pairs from `Dataset/` |
| **E6** self-audited faithfulness | ✅ yes (proxy) | `python evaluation/rag_self_faithfulness.py --auditor gemma` | same critic judges itself — declare as self-audit in paper |
| **E6'** cross-audited faithfulness | ✅ yes (stronger) | `python evaluation/rag_self_faithfulness.py --auditor gemini` | Gemini as external judge, critic-on-gemma as defendant |
| **E7** critic-on vs critic-off blind human | ❌ needs humans | — | 3 raters, Krippendorff's α |
| **E8–E11** analytics invariants | ✅ yes, CI-safe | `python evaluation/synthetic_user_sim.py` | — |
| **E12–E14** learning-outcome A/B | ❌ blocked | — | IRB, N ≥ 40 per arm |

Executive summary of what is **runnable right now** against the
currently deployed `gemma3-critic-v3-new` + Gemini fallback stack:

- **E1 + E2 + E4 + E6/E6'** on the existing 25-question test bank
  (20 in-scope Python/Data-Science + 5 adversarial OOS).
- **E8–E11** offline analytics (all passing as of commit `bc0985d`).

What blocks the other tiers is **data labelling** (E3, E5), **IRB**
(E12–E14), or **model training** (E4b); none is a code gap.

---

## Tier 1 — Critic model evaluation

### E1 — Validation accuracy on held-out document QA

- **RQ1.** Can a 4B parameter LoRA-fine-tuned SLM critic match or
  approach the accuracy of a frontier cloud LLM at document-grounding
  validation?
- **H1.** The v3 Gemma critic achieves ≥ 90% of Gemini 2.5's F1 on
  (question, excerpt) grounding, at ≥ 10× lower latency and zero
  per-call cost.
- **Design.** Held-out set of 200 labelled (question, excerpt, label)
  triples, stratified 50% in-scope / 50% out-of-scope, disjoint from
  the fine-tuning split.
- **Metrics.** Accuracy, precision, recall, F1 on the binary
  `confidence ≥ 85 ⇒ grounded` decision. Latency p50 / p95.
- **Statistics.** McNemar's test for paired proportions between
  {v1, v2, v3, Gemini} on the same triples.
- **Success.** v3 F1 ≥ 0.90 × Gemini F1.
- **Entry point.** `python evaluate.py --backends gemini gemma`.

### E2 — Confidence calibration

- **RQ2.** Are the critic's confidence scores *calibrated*, i.e. when it
  says 85% confident, is the excerpt actually grounded 85% of the time?
- **H2.** Expected Calibration Error (ECE) of v3 ≤ 0.08; reliability
  diagram lies on the identity line ± 10 pp.
- **Metrics.** ECE across 10 equal-width bins; Brier score; reliability
  diagram.
- **Statistics.** Bootstrap 95% CI on ECE (10,000 resamples).
- **Success.** ECE ≤ 0.08 **and** monotonic reliability curve (no
  confidence band where accuracy drops below the previous band).
- **Entry point.** `python evaluation/critic_calibration.py`.

### E3 — Scope-classification robustness

- **RQ3.** Does the dual-gate (similarity ≥ 0.20 **and** confidence > 85%)
  correctly reject out-of-scope questions without over-rejecting
  in-scope ones?
- **H3.** Dual-gate F1 > both single gates on an adversarial out-of-scope
  suite.
- **Design.** 100 in-scope + 100 adversarial out-of-scope queries
  (near-topic distractors). Ablation: similarity-only, confidence-only,
  dual-gate.
- **Metrics.** Precision / recall / F1 of "out-of-scope" as positive class.
- **Statistics.** Bootstrap 95% CI on F1; paired ablation contrast.
- **Success.** Dual-gate F1 ≥ max(single-gate F1) + 0.05.

### E4 — Cross-backend agreement (κ)

- **RQ4.** How often do v3 and Gemini agree on the pass/fail gate, and
  where do they disagree?
- **Metric.** Cohen's κ between v3 and Gemini binary decisions on the
  E1 held-out set. Qualitative: inspect 25 random disagreements.
- **Success.** κ ≥ 0.70. Documented failure modes for the remainder.
- **Entry point.** `python evaluate.py --backends gemini gemma` (uses
  existing agreement computation) + manual inspection.

### E4b — Qwen critic head-to-head (future)

Once the planned Qwen fine-tune lands, re-run E1–E4 with three backends
(Gemma-v3, Qwen-ft, Gemini). Report in the paper as an ablation of
base-model choice on identical LoRA data (`training_data_v3.jsonl`).

| Backend | Base params | Training data | Training tokens | ECE | F1 | Latency p50 |
|---------|-------------|---------------|-----------------|-----|----|-------------|
| gemma3-critic-v3 | 4B | `training_data_v3.jsonl` | ~1.2M | _TBD_ | _TBD_ | _TBD_ |
| qwen-critic-v1 | 4B | **same data** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Gemini 2.5 Flash | cloud | — | — | _TBD_ | _TBD_ | _TBD_ |

The Qwen run **must** use the same data split, LoRA rank/alpha, and
epoch count as the Gemma run for the comparison to be publishable.

---

## Tier 2 — RAG pipeline evaluation

### E5 — Retrieval quality

- **RQ5.** Does the Actor retrieve the gold-labelled chunk in the top-k?
- **Metrics.** Recall@k (k ∈ {1, 3, 5}), Mean Reciprocal Rank, nDCG@5.
- **Design.** 150 (question, gold-chunk-id) pairs constructed from
  existing session materials by manual labelling.
- **Entry point.** `python evaluation/retrieval_metrics.py`.

### E6 — Answer faithfulness (citation grounding)

- **RQ6.** Is every sentence of a "document-source" answer actually
  supported by the cited chunks?
- **Procedure.** For each answer, split into sentences. For each
  sentence, re-run `critic.validate(sentence, cited_excerpt)`. Flag any
  sentence whose max confidence across cited chunks falls below 60%.
- **Metrics.** Per-answer faithfulness rate = (supported sentences) /
  (total sentences). Overall mean and distribution.
- **Success.** Mean faithfulness ≥ 0.92; no answer below 0.70.
- **Entry point.** `python evaluation/rag_faithfulness.py`.

### E7 — Ablation: Critic on vs off

- **RQ7.** Does the Critic gate reduce hallucination vs the
  retrieve-and-quote baseline?
- **Design.** Same 200 questions; compare {retrieve-only, retrieve +
  critic} on E6's faithfulness metric plus a blind human rating of
  answer quality (3-point Likert, n = 3 raters).
- **Statistics.** Paired Wilcoxon signed-rank test; inter-rater Krippendorff's α.
- **Success.** Critic-on faithfulness ≥ retrieve-only + 0.10 **and**
  human quality rating higher at p < 0.05.

---

## Tier 3 — Offline evaluation of adaptive-analytics modules

These are deterministic or near-deterministic. They validate that the
**mechanics** behave as advertised, independent of any human learner.
They run fast, should be in CI, and have a concrete harness at
`evaluation/synthetic_user_sim.py`.

### E8 — Ebbinghaus decay monotonicity

- **RQ8.** Does `_apply_decay` produce a score that is (a) monotonically
  non-increasing in `days_since_assessed`, and (b) bounded below by
  `MASTERY_DECAY_FLOOR`?
- **Procedure.** Evaluate `decayed(raw=90, days=d)` for d ∈ [0, 365] at
  1-day granularity.
- **Assertions.** Δdecayed ≤ 0 everywhere; decayed ≥ FLOOR;
  decayed(d=0) = raw; half-life within ±10% of ln(2)/λ.
- **Success.** All assertions pass.

### E9 — Spaced-repetition schedule correctness

- **RQ9.** Given the Ebbinghaus model, does `review_scheduler` return a
  review date consistent with `d = ln(raw/T)/λ`?
- **Procedure.** Build synthetic skill records at known decay states
  and compare scheduler's `next_review_at` against the closed-form.
- **Metric.** Max absolute error across 1000 synthetic skills.
- **Success.** Max error < 0.5 days.

### E10 — ZPD mastery→Bloom mapping invariants

- **RQ10.** Does `_get_bloom_levels_for_zpd` honour the monotonicity
  requirement (higher mastery → higher Bloom level)?
- **Assertions.**
  - For mastery m1 < m2, the target Bloom window of m1 is weakly
    below m2 (no level drops as mastery rises).
  - For every mastery value, exactly two adjacent Bloom levels are
    returned.
  - All six Bloom levels are reachable across the mastery range
    [0, 100] × difficulty ∈ {beginner, intermediate, advanced}.
- **Success.** All three assertions hold for every m ∈ [0, 100] step 1.

### E11 — Knowledge-transfer Pearson recovery

- **RQ11.** Given a ground-truth linear relationship
  `session_score = α · prereq_mastery + β + ε`, can
  `knowledge_transfer` recover α (via Pearson r) within noise?
- **Procedure.** Synthesise 50 attempts per prereq with
  α ∈ {0.0, 0.5, 0.9} and Gaussian noise ε ~ N(0, σ²). Run
  `compute_transfer` and compare recovered r to ground truth.
- **Success.** |r_recovered − α| < 0.1 for σ ≤ 0.2.

---

## Tier 4 — End-to-end human learning outcomes

These are the experiments that let the paper claim *learning gains*,
not just *system correctness*. They require IRB approval and an
informed-consent flow.

### E12 — A/B of Ebbinghaus-decayed path adaptation

- **RQ12.** Do learners on decay-aware adaptation retain more a week
  later than learners on a raw-score baseline?
- **Design.** Between-subjects, N ≥ 40 per arm, 10 sessions each,
  retention measured by a blind quiz 7 days after completion.
- **Arms.**
  - Control: path adapter uses raw mastery (no decay on read).
  - Treatment: path adapter uses decayed mastery (current system).
- **Statistics.** Mixed-effects model (session number as within-subject
  factor, arm as between-subject); effect-size Cohen's d on 7-day
  retention.
- **Success.** Treatment arm retention higher by ≥ 5 pp, p < 0.05,
  d ≥ 0.3.

### E13 — A/B of ZPD-targeted quizzes

- **RQ13.** Do Bloom-ZPD-targeted quizzes accelerate mastery compared
  to difficulty-uniform quizzes?
- **Design.** Same as E12; arms differ only in `question_generator`:
  uniform vs ZPD-targeted.
- **Primary metric.** Sessions to reach mastered (> 85%) per skill.
- **Secondary.** Self-reported engagement (SEI short form), calibration
  gap.
- **Statistics.** Survival analysis (Cox proportional hazards) on
  sessions-to-mastery; bootstrap 95% CI on median.
- **Success.** Hazard ratio ≥ 1.3 in treatment (faster mastery).

### E14 — Calibration feedback loop

- **RQ14.** Does surfacing the calibration scatter-plot reduce
  over-/under-confidence over 10 sessions?
- **Design.** Pre/post-intervention within-subject, N ≥ 30.
- **Metric.** Mean |gap| on the last 3 sessions vs first 3.
- **Statistics.** Paired Wilcoxon; Cohen's d on gap change.
- **Success.** |gap| reduction ≥ 5 pp, p < 0.05.

---

## Reporting template

Every experiment in the paper must include:

1. **Pre-registration** — RQ, H, metric, stopping rule, primary vs
   secondary outcomes (for E12–E14).
2. **Data statement** — size, source, labelling procedure, inter-rater
   κ where applicable.
3. **Statistics** — test, assumptions checked (normality, equal
   variances), effect size with 95% CI, not just a p-value.
4. **Ablation table** — every claimed component contributes
   measurably; drop it, show the drop.
5. **Negative results** — hypotheses that *didn't* survive are
   reported, not suppressed.

---

## Ablation matrix (summary to include in the paper)

| # | Component ablated | Expected metric drop |
|---|-------------------|----------------------|
| A1 | Replace fine-tuned critic with base Gemma (no LoRA) | F1 in E1 |
| A2 | Replace dual-gate with similarity-only | F1 in E3 |
| A3 | Turn off Ebbinghaus decay (use raw) | 7-day retention in E12 |
| A4 | Replace ZPD targeting with uniform difficulty | sessions-to-mastery in E13 |
| A5 | Remove citation rendering from UI | faithfulness-audit completion rate |
| A6 | Remove calibration panel from UI | calibration improvement in E14 |
| A7 | Replace Pearson transfer with uniform remediation | prerequisite-violation rate |
| A8 | Replace Kahn topological reorder with submission order | prerequisite-violation rate |

Each ablation is a one-flag config change; none require code rewrites.

---

## Datasets

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| `training_data_v3.jsonl` | LoRA fine-tuning | ~800 | Curated (see `generate_training_data.py`) |
| `held_out_qa.jsonl` | E1, E2, E4 | 200 | **To label.** 50% in-scope from Dataset/, 50% OOS |
| `retrieval_gold.jsonl` | E5 | 150 | **To label.** (Q, gold-chunk-id) per session |
| `adversarial_oos.jsonl` | E3 | 100 | **To construct.** Near-topic distractors |
| `sentence_faithfulness_set.jsonl` | E6, E7 | 200 answers | Generated by current system + manual check |
| IRB user-study cohort | E12–E14 | N ≥ 120 | **To recruit.** University course volunteers |

All labelling uses a 2-annotator + tie-breaker protocol with
Krippendorff's α reported.

---

## Computational cost

Report in the paper:

- Total GPU-hours for fine-tuning (per backend).
- Latency budget per route: Actor retrieve, Critic validate, Critic
  generate; p50 and p95.
- Token cost per session end-to-end on (a) all-local stack, (b)
  Gemini-fallback stack.

This section is what reviewers check to assess reproducibility and
whether the "efficient fine-tuning" story holds up.

---

## Threats to validity

- **Internal.** Label noise on OOS / faithfulness sets → mitigated by
  dual-annotator labelling.
- **External.** Only one subject domain (career paths); generalisation
  to K-12 or medical training not tested.
- **Construct.** "Mastery" is proxied by quiz percentage; retention
  test (E12) partially addresses this but is still quiz-based.
- **Statistical.** Multiple comparisons across E1–E14 → apply
  Benjamini-Hochberg FDR control on the family of primary hypotheses.

---

## Running the offline tiers

```bash
# Tier 1 — side-by-side descriptive metrics (existing, unchanged)
python evaluate.py --backends gemini gemma --out results_t1_legacy.json

# Tier 1 — publication-grade metrics (F1 + ECE + Brier + Cohen's κ + McNemar)
python evaluation/tier1_critic_benchmark.py --backends gemma gemini

# Tier 2 — self-audited faithfulness (gemma judges gemma)
python evaluation/rag_self_faithfulness.py --critic gemma --auditor gemma

# Tier 2 — cross-audited faithfulness (gemini judges gemma) — stronger
python evaluation/rag_self_faithfulness.py --critic gemma --auditor gemini

# Tier 3 — deterministic, CI-safe
python evaluation/synthetic_user_sim.py
```

Tier 4 requires IRB clearance and is run outside CI.

### Qwen drop-in (planned)

When `qwen-critic-v1` (LoRA-fine-tuned on the same
`training_data_v3.jsonl` as Gemma-v3) is registered in `critic.py` via
`create_critic("qwen")`, re-run the Tier-1 command with three backends:

```bash
python evaluation/tier1_critic_benchmark.py --backends gemma qwen gemini
```

The same `tier1_report.json` will populate a third backend column and
the pairwise agreement section will report κ for (gemma, qwen),
(gemma, gemini), (qwen, gemini). This is exactly the ablation the paper
needs to separate *architecture effect* (Qwen vs Gemma) from *training
data effect* (both on the same LoRA data).
