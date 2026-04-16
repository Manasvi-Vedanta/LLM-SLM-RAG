"""
knowledge_transfer.py – Cross-session knowledge transfer metrics
------------------------------------------------------------------
Measures how well prerequisite mastery predicts performance on
dependent sessions. For each session with a completed quiz, pairs
its score with the average mastery of its prerequisites, then
computes correlation and per-prerequisite transfer strength.

Use cases:
  - Flag broken prerequisite chains (low prereq mastery → low score)
  - Identify well-transferring skills (high prereq mastery → high score)
  - Validate that the path's dependency graph reflects actual learning

Outputs feed into path adaptation and the mastery dashboard.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import database

logger = logging.getLogger(__name__)


@dataclass
class TransferPair:
    """A single (prereq_mastery_avg, session_score) observation."""
    session_number: int
    session_title: str
    prerequisites: list[str]
    prereq_mastery_avg: float
    session_score: float
    attempt_number: int


@dataclass
class PrereqTransfer:
    """Transfer strength for a single prerequisite skill."""
    prerequisite: str
    dependent_sessions: list[int]
    sample_count: int
    prereq_mastery_avg: float
    dependent_score_avg: float
    transfer_score: float    # 0-100: how much mastery translates to dependent performance


@dataclass
class TransferReport:
    pairs: list[TransferPair] = field(default_factory=list)
    per_prerequisite: list[PrereqTransfer] = field(default_factory=list)
    correlation: float | None = None    # Pearson r
    sample_count: int = 0
    weak_chains: list[str] = field(default_factory=list)
    strong_chains: list[str] = field(default_factory=list)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient. Returns None for n<3 or zero variance."""
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def compute_transfer(
    user_id: int,
    path_id: int,
    learning_path: list[dict],
) -> TransferReport:
    """Compute knowledge-transfer metrics across sessions.

    For each session with ≥1 quiz attempt, pair the attempt score with
    the average mastery of its prerequisites at the time.
    """
    report = TransferReport()

    # Snapshot current mastery (decayed) by skill
    mastery_rows = database.get_skill_mastery(user_id, path_id)
    mastery_map = {r["skill_label"]: r["mastery_score"] for r in mastery_rows}

    # For each session: look up the latest quiz attempt and its prerequisites
    pairs: list[TransferPair] = []

    for session in learning_path:
        sn = session.get("session_number", 0)
        title = session.get("title", "")
        prereqs = session.get("prerequisites", [])

        if not prereqs:
            continue

        attempts = database.get_quiz_attempts(path_id, sn, user_id)
        if not attempts:
            continue

        # Use best attempt to avoid first-try noise biasing the signal
        best = max(attempts, key=lambda a: a.get("percentage", 0))
        score = float(best.get("percentage", 0) or 0)

        prereq_scores = [mastery_map.get(p) for p in prereqs]
        prereq_scores = [s for s in prereq_scores if s is not None]
        if not prereq_scores:
            continue

        pairs.append(TransferPair(
            session_number=sn,
            session_title=title,
            prerequisites=prereqs,
            prereq_mastery_avg=sum(prereq_scores) / len(prereq_scores),
            session_score=score,
            attempt_number=best.get("attempt_number", 1),
        ))

    report.pairs = pairs
    report.sample_count = len(pairs)

    if len(pairs) >= 3:
        xs = [p.prereq_mastery_avg for p in pairs]
        ys = [p.session_score for p in pairs]
        report.correlation = _pearson(xs, ys)

    # Per-prerequisite transfer strength:
    # for each prereq skill X, look at all sessions that depend on X and
    # compare: how well does mastery of X predict success in those sessions?
    prereq_index: dict[str, list[TransferPair]] = {}
    for p in pairs:
        for prereq in p.prerequisites:
            prereq_index.setdefault(prereq, []).append(p)

    for prereq, dep_pairs in prereq_index.items():
        prereq_mastery = mastery_map.get(prereq)
        if prereq_mastery is None:
            continue

        dep_score_avg = sum(dp.session_score for dp in dep_pairs) / len(dep_pairs)

        # Transfer score: min of prereq mastery and dependent performance.
        # A skill "transferred" only as far as the downstream actually achieved.
        transfer_score = min(prereq_mastery, dep_score_avg)

        tr = PrereqTransfer(
            prerequisite=prereq,
            dependent_sessions=[dp.session_number for dp in dep_pairs],
            sample_count=len(dep_pairs),
            prereq_mastery_avg=round(prereq_mastery, 1),
            dependent_score_avg=round(dep_score_avg, 1),
            transfer_score=round(transfer_score, 1),
        )
        report.per_prerequisite.append(tr)

        # Flag weak/strong transfer chains
        gap = prereq_mastery - dep_score_avg
        if prereq_mastery >= 70 and dep_score_avg < 50:
            # High mastery of prereq but weak dependent performance → broken transfer
            report.weak_chains.append(
                f"{prereq} ({prereq_mastery:.0f}%) -> downstream avg {dep_score_avg:.0f}%"
            )
        elif prereq_mastery >= 70 and dep_score_avg >= 70:
            report.strong_chains.append(
                f"{prereq} ({prereq_mastery:.0f}%) -> downstream avg {dep_score_avg:.0f}%"
            )

    # Sort per-prereq by transfer strength ascending (weakest first)
    report.per_prerequisite.sort(key=lambda t: t.transfer_score)

    logger.info(
        "Knowledge transfer: %d pairs, correlation=%s, weak chains=%d, strong chains=%d",
        report.sample_count,
        f"{report.correlation:.2f}" if report.correlation is not None else "n/a",
        len(report.weak_chains), len(report.strong_chains),
    )

    return report
