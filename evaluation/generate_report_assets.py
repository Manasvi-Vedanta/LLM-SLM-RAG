"""
Generate report-ready tables (.md + .tex) and figures (.png) from the
evaluation result JSONs. Output goes to evaluation/results/report_assets/.

    python evaluation/generate_report_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "report_assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titlepad": 10,
    "axes.labelsize": 11,
    "axes.labelpad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.15,
    "figure.constrained_layout.w_pad": 0.15,
})


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# -------------------------------------------------------------------- #
# Data
# -------------------------------------------------------------------- #
tier1 = load("tier1_with_base_ablation.json")
tier2_self = load("tier2_faithfulness.json")
tier2_cross = load("tier2_cross_audit_gemini.json")
tier2_simonly = load("tier2_similarity_only_full.json")
tier3 = load("tier3_report.json")
tier4 = load("tier4_synthetic_rct.json")

backends = tier1["backends"]
GEMMA = backends["gemma"]
QWEN = backends["qwen"]
BASE = backends["base"]


# -------------------------------------------------------------------- #
# Figure 1 — Tier 1 critic comparison (3 backends, 4 metrics)
# -------------------------------------------------------------------- #
def fig_tier1_critics():
    labels = ["gemma\n(LoRA)", "qwen\n(LoRA)", "base\n(no LoRA)"]
    accs = [b["E1_classification"]["accuracy"] for b in (GEMMA, QWEN, BASE)]
    f1s  = [b["E1_classification"]["f1"]       for b in (GEMMA, QWEN, BASE)]
    eces = [b["E2_calibration"]["ece"]         for b in (GEMMA, QWEN, BASE)]
    lats = [b["latency"]["p50_s"]              for b in (GEMMA, QWEN, BASE)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    colors = ["#2a7", "#27a", "#a72"]

    for ax, vals, title, ylabel, lower_better in [
        (axes[0], accs, "Accuracy", "fraction", False),
        (axes[1], f1s,  "F1 score", "F1", False),
        (axes[2], eces, "ECE (calibration error)", "ECE", True),
        (axes[3], lats, "Latency (p50)", "seconds", True),
    ]:
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6,
                      width=0.65)
        ax.set_title(title + ("  ↓" if lower_better else "  ↑"))
        ax.set_ylabel(ylabel, labelpad=10)
        ymax = max(vals) * 1.30
        for bar, v in zip(bars, vals):
            offset = ymax * 0.025
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:.2f}" if v < 10 else f"{v:.1f}s",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", pad=4)
        ax.margins(x=0.18)

    fig.suptitle("Tier 1 — critic backend comparison (n = 25 questions)",
                 fontsize=14, fontweight="semibold")
    fig.savefig(OUT / "fig1_tier1_critics.png")
    plt.close(fig)


# -------------------------------------------------------------------- #
# Figure 2 — Reliability diagrams (calibration) for the three critics
# -------------------------------------------------------------------- #
def fig_reliability():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    triples = [
        ("gemma (LoRA)", GEMMA, "#2a7"),
        ("qwen (LoRA)",  QWEN,  "#27a"),
        ("base (no LoRA)", BASE, "#a72"),
    ]
    for ax, (name, b, color) in zip(axes, triples):
        rel = b["E2_calibration"]["reliability"]
        confs = [r["avg_conf"] for r in rel if r["count"] > 0]
        accs  = [r["accuracy"] for r in rel if r["count"] > 0]
        counts = [r["count"]   for r in rel if r["count"] > 0]
        sizes = [40 + 50 * c for c in counts]

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect")
        ax.scatter(confs, accs, s=sizes, color=color, edgecolor="black",
                   linewidth=0.8, alpha=0.85, zorder=3, label=name)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("predicted confidence", labelpad=8)
        ax.set_ylabel("empirical accuracy", labelpad=8)
        ax.set_title(f"{name}\nECE = {b['E2_calibration']['ece']:.3f}",
                     pad=12)
        ax.set_aspect("equal", "box")

    fig.suptitle("Tier 1 (E2) — reliability diagrams  (bubble size ∝ bin count)",
                 fontsize=14, fontweight="semibold")
    fig.savefig(OUT / "fig2_reliability.png")
    plt.close(fig)


# -------------------------------------------------------------------- #
# Figure 3 — Tier 2 faithfulness (self vs cross-audit vs sim-only)
# -------------------------------------------------------------------- #
def fig_tier2_faithfulness():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rows = [
        ("E6 self\n(gemma → gemma)",         tier2_self["summary"]["mean_faithfulness"],
         tier2_self["summary"]["n_document_answers"], "#2a7"),
        ("E6' cross\n(gemini → gemma)",      tier2_cross["summary"]["mean_faithfulness"],
         tier2_cross["summary"]["n_document_answers"], "#27a"),
        ("A2 sim-only\n(conf gate off)",     tier2_simonly["summary"]["mean_faithfulness"],
         tier2_simonly["summary"]["n_document_answers"], "#c63"),
    ]
    labels = [r[0] for r in rows]
    means  = [r[1] for r in rows]
    ndocs  = [r[2] for r in rows]
    cols   = [r[3] for r in rows]

    bars = ax.bar(labels, means, color=cols, edgecolor="black", linewidth=0.6,
                  width=0.55)
    for bar, m, n in zip(bars, means, ndocs):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.025,
                f"{m:.3f}\n(n_doc={n})", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0.0, 1.30)
    ax.set_ylabel("mean sentence-level faithfulness", labelpad=10)
    ax.set_title("Tier 2 — faithfulness across audit configurations",
                 fontsize=14, fontweight="semibold", pad=14)
    ax.axhline(0.7, ls="--", color="red", alpha=0.5, label="0.70 alarm threshold")
    ax.legend(loc="lower right", frameon=True)
    ax.tick_params(axis="x", pad=4)
    fig.savefig(OUT / "fig3_tier2_faithfulness.png")
    plt.close(fig)


# -------------------------------------------------------------------- #
# Figure 4 — A2 routing leakage (OOS masquerading)
# -------------------------------------------------------------------- #
def fig_a2_routing():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cats = ["Dual gate\n(production)", "Similarity-only\n(conf gate off)"]
    in_scope = [20, 20]
    oos      = [0, 5]
    x = np.arange(len(cats))
    ax.bar(x, in_scope, color="#2a7", edgecolor="black",
           label="in-scope routed as document", width=0.55)
    ax.bar(x, oos, bottom=in_scope, color="#c33", edgecolor="black",
           label="OOS leaked as document", width=0.55)
    for i, (a, b) in enumerate(zip(in_scope, oos)):
        ax.text(i, a + b + 0.6, f"{a + b}/25", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("answers routed as document", labelpad=10)
    ax.set_ylim(0, 32)
    ax.set_title("A2 — confidence gate prevents OOS leakage",
                 fontsize=14, fontweight="semibold", pad=14)
    ax.legend(loc="upper left", frameon=True)
    ax.tick_params(axis="x", pad=4)
    fig.savefig(OUT / "fig4_a2_routing.png")
    plt.close(fig)


# -------------------------------------------------------------------- #
# Figure 5 — Tier 4 RCTs
# -------------------------------------------------------------------- #
def fig_tier4_rct():
    res = {r["name"]: r for r in tier4["results"]}
    e12 = res["E12_decay_vs_raw_retention"]
    e13 = res["E13_zpd_vs_uniform_sessions_to_mastery"]
    e14 = res["E14_calibration_feedback_reduces_gap"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # E12 — retention
    a = axes[0]
    vals = [e12["control_mean_retention"], e12["treatment_mean_retention"]]
    bars = a.bar(["Control\n(raw mastery)", "Treatment\n(decay-aware)"],
                 vals, color=["#aaa", "#2a7"], edgecolor="black", width=0.55)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width()/2, v + 1.5,
               f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    a.set_ylabel("7-day retention (%)", labelpad=10)
    a.set_ylim(0, 78)
    a.set_title(f"E12 — retention\n"
                f"Δ = +{e12['delta_pp']:.2f} pp · d = {e12['cohens_d']:.2f} · p < 0.001",
                pad=12)
    a.tick_params(axis="x", pad=4)

    # E13 — time-to-mastery
    a = axes[1]
    vals = [e13["control_mean_sessions"], e13["treatment_mean_sessions"]]
    bars = a.bar(["Control\n(uniform Bloom)", "Treatment\n(ZPD Bloom)"],
                 vals, color=["#aaa", "#27a"], edgecolor="black", width=0.55)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width()/2, v + 0.18,
               f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    a.set_ylabel("sessions to latent mastery", labelpad=10)
    a.set_ylim(0, max(vals) * 1.30)
    a.set_title(f"E13 — time-to-mastery\n"
                f"Δ = {e13['delta_sessions']:+.2f} sessions · "
                f"HR = {e13['hazard_ratio']:.0f} · p < 0.001",
                pad=12)
    a.tick_params(axis="x", pad=4)

    # E14 — calibration (has a negative control + a positive treatment)
    a = axes[2]
    vals = [e14["control_gap_reduction_pp"], e14["treatment_gap_reduction_pp"]]
    bars = a.bar(["Control\n(no feedback)", "Treatment\n(calibration)"],
                 vals, color=["#aaa", "#c63"], edgecolor="black", width=0.55)
    for b, v in zip(bars, vals):
        offset = 0.30 if v >= 0 else -0.55
        va = "bottom" if v >= 0 else "top"
        a.text(b.get_x() + b.get_width()/2, v + offset,
               f"{v:+.2f}", ha="center", va=va, fontsize=11, fontweight="bold")
    a.set_ylabel("|gap| reduction, first 3 → last 3 (pp)", labelpad=10)
    a.set_ylim(-2, 9.5)
    a.set_title(f"E14 — calibration\n"
                f"Δ = +{e14['delta_reduction_pp']:.2f} pp · "
                f"d = {e14['cohens_d']:.2f} · p < 0.001",
                pad=12)
    a.axhline(0, color="black", lw=0.6)
    a.tick_params(axis="x", pad=4)

    fig.suptitle("Tier 4 — synthetic-learner RCTs  (N = 1000 per arm · seed = 42)",
                 fontsize=14, fontweight="semibold")
    fig.savefig(OUT / "fig5_tier4_rct.png")
    plt.close(fig)


# -------------------------------------------------------------------- #
# Figure 6 — Tier 3 invariants (pass/fail summary)
# -------------------------------------------------------------------- #
def fig_tier3_invariants():
    fig, ax = plt.subplots(figsize=(10, 4))
    rows = tier3["results"]
    nice = {
        "E8_ebbinghaus_monotonicity":      "E8 · Ebbinghaus monotonicity",
        "E9_spaced_repetition_closed_form":"E9 · Spaced-repetition closed form",
        "E10_zpd_invariants":              "E10 · ZPD Bloom invariants",
        "E11_pearson_recovery":            "E11 · Pearson recovery",
    }
    names = [nice.get(r["name"], r["name"].replace("_", " ")) for r in rows]
    passed = [r["passed"] for r in rows]
    colors = ["#2a7" if p else "#c33" for p in passed]
    ax.barh(names, [1] * len(names), color=colors, edgecolor="black", height=0.55)
    for i, p in enumerate(passed):
        ax.text(0.5, i, "PASS" if p else "FAIL",
                ha="center", va="center", color="white",
                fontweight="bold", fontsize=12)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.tick_params(axis="y", pad=6)
    ax.set_title("Tier 3 — analytical invariants  (E8 – E11)",
                 fontsize=14, fontweight="semibold", pad=12)
    fig.savefig(OUT / "fig6_tier3_invariants.png")
    plt.close(fig)


# -------------------------------------------------------------------- #
# Markdown + LaTeX tables
# -------------------------------------------------------------------- #
def _e1(b): return b["E1_classification"]
def _e2(b): return b["E2_calibration"]
def _lat(b): return b["latency"]


def write_tables():
    md = []
    tex = []

    # ---- Table 1: Tier 1 critic comparison ----
    md.append("## Table 1 — Tier 1 critic comparison (n = 25)\n")
    md.append("| Backend | Accuracy | F1 | Precision | Recall | Specificity | ECE ↓ | Brier ↓ | p50 / p95 latency |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, b, label in [("gemma3-critic-v3-new (LoRA)", GEMMA, "gemma"),
                            ("qwen-critic-v1 (LoRA)", QWEN, "qwen"),
                            ("gemma3:4b (no LoRA, A1 ablation)", BASE, "base")]:
        e1, e2, lat = _e1(b), _e2(b), _lat(b)
        md.append(f"| {name} | {e1['accuracy']:.2f} | {e1['f1']:.3f} | "
                  f"{e1['precision']:.3f} | {e1['recall']:.3f} | {e1['specificity']:.3f} | "
                  f"{e2['ece']:.3f} | {e2['brier']:.3f} | "
                  f"{lat['p50_s']:.1f}s / {lat['p95_s']:.1f}s |")
    md.append("")

    tex.append(r"\begin{table}[t]\centering")
    tex.append(r"\caption{Tier 1 critic comparison (n=25). LoRA fine-tuning lifts Gemma F1 by 38\,pp and reduces ECE 4.2$\times$ vs. the base model.}")
    tex.append(r"\label{tab:tier1}")
    tex.append(r"\begin{tabular}{lrrrrrrrl}\toprule")
    tex.append(r"Backend & Acc & F1 & Prec & Rec & Spec & ECE$\downarrow$ & Brier$\downarrow$ & p50/p95 lat \\\midrule")
    for name, b in [("gemma3-critic-v3-new (LoRA)", GEMMA),
                     ("qwen-critic-v1 (LoRA)", QWEN),
                     ("gemma3:4b (no LoRA, A1)", BASE)]:
        e1, e2, lat = _e1(b), _e2(b), _lat(b)
        tex.append(f"{name} & {e1['accuracy']:.2f} & {e1['f1']:.3f} & "
                   f"{e1['precision']:.3f} & {e1['recall']:.3f} & {e1['specificity']:.3f} & "
                   f"{e2['ece']:.3f} & {e2['brier']:.3f} & "
                   f"{lat['p50_s']:.1f}s/{lat['p95_s']:.1f}s \\\\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    tex.append("")

    # ---- Table 2: Tier 2 faithfulness ----
    md.append("## Table 2 — Tier 2 faithfulness (E6, E6', A2)\n")
    md.append("| Configuration | n_doc | Mean | Median | Min | Answers < 0.7 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for label, src in [
        ("E6 self-audit (gemma → gemma)", tier2_self["summary"]),
        ("E6' cross-audit (gemini → gemma)", tier2_cross["summary"]),
        ("A2 similarity-only (conf gate off, 25Q)", tier2_simonly["summary"]),
    ]:
        md.append(f"| {label} | {src['n_document_answers']} | "
                  f"{src['mean_faithfulness']:.3f} | {src['median_faithfulness']:.3f} | "
                  f"{src['min_faithfulness']:.3f} | {src['answers_below_0_7']} |")
    md.append("")

    tex.append(r"\begin{table}[t]\centering")
    tex.append(r"\caption{Faithfulness audits. The cross-audit (E6$'$) corroborates the self-audit; A2 demonstrates that disabling the confidence gate does not collapse mean faithfulness but enables OOS routing leakage (Table~\ref{tab:a2}).}")
    tex.append(r"\label{tab:tier2}")
    tex.append(r"\begin{tabular}{lrrrrr}\toprule")
    tex.append(r"Config & $n_{\text{doc}}$ & Mean & Med & Min & $<0.7$ \\\midrule")
    for label, src in [
        ("E6 self (gemma$\\to$gemma)", tier2_self["summary"]),
        ("E6$'$ cross (gemini$\\to$gemma)", tier2_cross["summary"]),
        ("A2 sim-only (conf off)", tier2_simonly["summary"]),
    ]:
        tex.append(f"{label} & {src['n_document_answers']} & "
                   f"{src['mean_faithfulness']:.3f} & {src['median_faithfulness']:.3f} & "
                   f"{src['min_faithfulness']:.3f} & {src['answers_below_0_7']} \\\\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    tex.append("")

    # ---- Table 3: A2 routing ----
    md.append("## Table 3 — A2 routing correctness (25Q sweep: 20 in-scope + 5 OOS)\n")
    md.append("| Configuration | Routed as document | OOS leaked | Routing correct? |")
    md.append("|---|---:|---:|:---:|")
    md.append("| Dual gate (sim ≥ 0.20 AND conf ≥ 85) | 20 / 25 | 0 / 5 | ✅ |")
    md.append("| Similarity-only (conf gate off) | 25 / 25 | 5 / 5 | ❌ |")
    md.append("")

    tex.append(r"\begin{table}[t]\centering")
    tex.append(r"\caption{A2 ablation: the confidence gate is the routing invariant that prevents OOS questions from being answered as document-grounded.}")
    tex.append(r"\label{tab:a2}")
    tex.append(r"\begin{tabular}{lrrc}\toprule")
    tex.append(r"Config & Doc-routed & OOS leaked & Correct? \\\midrule")
    tex.append(r"Dual gate & 20/25 & 0/5 & \checkmark \\")
    tex.append(r"Sim-only & 25/25 & 5/5 & $\times$ \\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    tex.append("")

    # ---- Table 4: Tier 3 invariants ----
    md.append("## Table 4 — Tier 3 analytical invariants\n")
    md.append("| Test | Result | Key statistic |")
    md.append("|---|:---:|---|")
    invariant_summaries = {
        "E8_ebbinghaus_monotonicity":
            f"half-life error {tier3['results'][0]['details']['halflife_err_pct']:.2f}%, 0 monotonicity violations",
        "E9_spaced_repetition_closed_form":
            f"max error {tier3['results'][1]['details']['max_abs_error_pp']:.2f} pp over {tier3['results'][1]['details']['samples']} samples",
        "E10_zpd_invariants":
            f"all 6 Bloom levels reached, 0 monotonicity / adjacency violations",
        "E11_pearson_recovery":
            "α=0→r≈0; α=0.5→r=0.41; α=0.9→r=0.87 (max |err|=0.09)",
    }
    for r in tier3["results"]:
        md.append(f"| {r['name']} | {'✅' if r['passed'] else '❌'} | {invariant_summaries[r['name']]} |")
    md.append("")

    tex.append(r"\begin{table}[t]\centering")
    tex.append(r"\caption{Tier 3 analytical invariants. All four pass.}")
    tex.append(r"\label{tab:tier3}")
    tex.append(r"\begin{tabular}{llp{8cm}}\toprule")
    tex.append(r"Test & Pass & Key statistic \\\midrule")
    for r in tier3["results"]:
        tex.append(f"{r['name'].replace('_', ' ')} & "
                   f"{'\\checkmark' if r['passed'] else '$\\times$'} & "
                   f"{invariant_summaries[r['name']]} \\\\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    tex.append("")

    # ---- Table 5: Tier 4 RCTs ----
    md.append("## Table 5 — Tier 4 synthetic-learner RCTs (N = 1000 / arm)\n")
    md.append("| Exp | Contrast | Control | Treatment | Δ | Cohen's d | t | p |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in tier4["results"]:
        if "control_mean_retention" in r:
            ctl, trt = r["control_mean_retention"], r["treatment_mean_retention"]
            d = trt - ctl
            unit = "pp"
        elif "control_mean_sessions" in r:
            ctl, trt = r["control_mean_sessions"], r["treatment_mean_sessions"]
            d = trt - ctl
            unit = "sessions"
        else:
            ctl, trt = r["control_gap_reduction_pp"], r["treatment_gap_reduction_pp"]
            d = trt - ctl
            unit = "pp"
        t_or = r.get("t_stat")
        t_str = "—" if t_or is None else f"{t_or:.2f}"
        p_val = r.get("p_value", r.get("sign_test_p_value", 1.0))
        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        md.append(f"| {r['name'].split('_')[0]} | "
                  f"{r['name'].replace('_', ' ').split(' ', 1)[1]} | "
                  f"{ctl:.2f} | {trt:.2f} | {d:+.2f} {unit} | "
                  f"{r['cohens_d']:.2f} | {t_str} | {p_str} |")
    md.append("")

    tex.append(r"\begin{table}[t]\centering")
    tex.append(r"\caption{Synthetic-learner RCTs (N=1000/arm, seed=42). All three pre-registered contrasts are directionally correct and significant; E13 and E14 also clear pre-registered effect-size thresholds.}")
    tex.append(r"\label{tab:tier4}")
    tex.append(r"\begin{tabular}{llrrrrr}\toprule")
    tex.append(r"Exp & Contrast & Control & Treatment & $\Delta$ & $d$ & $p$ \\\midrule")
    for r in tier4["results"]:
        if "control_mean_retention" in r:
            ctl, trt, unit = r["control_mean_retention"], r["treatment_mean_retention"], "pp"
        elif "control_mean_sessions" in r:
            ctl, trt, unit = r["control_mean_sessions"], r["treatment_mean_sessions"], "sess"
        else:
            ctl, trt, unit = r["control_gap_reduction_pp"], r["treatment_gap_reduction_pp"], "pp"
        d = trt - ctl
        contrast = r["name"].replace("_", " ").split(" ", 1)[1]
        p_val = r.get("p_value", r.get("sign_test_p_value", 1.0))
        p_tex = r"$<\!0.001$" if p_val < 0.001 else f"{p_val:.3f}"
        tex.append(f"{r['name'].split('_')[0]} & {contrast} & "
                   f"{ctl:.2f} & {trt:.2f} & {d:+.2f}\\,{unit} & "
                   f"{r['cohens_d']:.2f} & {p_tex} \\\\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    tex.append("")

    # ---- Master headline ----
    md.insert(0, "# Report assets — tables\n")
    md.insert(1, "Auto-generated from `evaluation/results/*.json` by `evaluation/generate_report_assets.py`.\n")

    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "tables.tex").write_text("\n".join(tex), encoding="utf-8")


# -------------------------------------------------------------------- #
def main():
    fig_tier1_critics()
    fig_reliability()
    fig_tier2_faithfulness()
    fig_a2_routing()
    fig_tier4_rct()
    fig_tier3_invariants()
    write_tables()
    print(f"Wrote assets to: {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
