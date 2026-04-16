"""
mastery_tracker.py – Skill mastery tracking and readiness checks
-----------------------------------------------------------------
Aggregates per-skill mastery scores, checks prerequisite readiness,
and provides summaries for path adaptation decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config
import database

logger = logging.getLogger(__name__)


# ── data structures ──────────────────────────────────────────────────

@dataclass
class SkillMastery:
    skill_label: str
    session_number: int
    mastery_score: float          # 0-100
    attempts_count: int
    status: str                   # "mastered", "review", "weak", "not_assessed"


@dataclass
class ReadinessResult:
    session_number: int
    is_ready: bool
    prerequisites: list[str]
    unmet_prerequisites: list[str]
    details: list[dict] = field(default_factory=list)
    # details: [{"skill": "Python", "score": 90, "status": "mastered"}]


@dataclass
class MasterySummary:
    total_skills: int
    mastered_count: int
    review_count: int
    weak_count: int
    not_assessed_count: int
    overall_percentage: float
    skills: list[SkillMastery] = field(default_factory=list)


# ── mastery classification ───────────────────────────────────────────

def _classify_mastery(score: float) -> str:
    """Classify a mastery score into a status label."""
    if score >= config.MASTERY_THRESHOLD_SKIP:
        return "mastered"
    elif score >= config.MASTERY_THRESHOLD_REVIEW:
        return "review"
    else:
        return "weak"


# ── public API ───────────────────────────────────────────────────────

def update_mastery(user_id: int, path_id: int, session_number: int,
                   skill_label: str, quiz_score: float) -> None:
    """Update mastery score for a skill after a quiz attempt."""
    database.upsert_skill_mastery(
        user_id=user_id,
        path_id=path_id,
        skill_label=skill_label,
        session_number=session_number,
        new_score=quiz_score,
    )
    logger.info("Updated mastery: %s = %.1f%% (path %d, session %d)",
                skill_label, quiz_score, path_id, session_number)


def get_skill_mastery(user_id: int, path_id: int) -> list[SkillMastery]:
    """Get all mastery records for a learning path."""
    rows = database.get_skill_mastery(user_id, path_id)
    return [
        SkillMastery(
            skill_label=r["skill_label"],
            session_number=r["session_number"],
            mastery_score=r["mastery_score"],
            attempts_count=r["attempts_count"],
            status=_classify_mastery(r["mastery_score"]),
        )
        for r in rows
    ]


def get_session_readiness(
    user_id: int,
    path_id: int,
    session_number: int,
    corrected_path: dict,
) -> ReadinessResult:
    """Check if prerequisite skills are mastered for a session.

    Looks up the session's prerequisites, checks each prerequisite
    skill's mastery score, and reports whether the user is ready.
    """
    sessions = corrected_path.get("learning_path", [])
    target_session = None
    for s in sessions:
        if s.get("session_number") == session_number:
            target_session = s
            break

    if not target_session:
        return ReadinessResult(
            session_number=session_number,
            is_ready=True,
            prerequisites=[],
            unmet_prerequisites=[],
        )

    prerequisites = target_session.get("prerequisites", [])
    if not prerequisites:
        return ReadinessResult(
            session_number=session_number,
            is_ready=True,
            prerequisites=[],
            unmet_prerequisites=[],
        )

    # Get all mastery records
    mastery_records = database.get_skill_mastery(user_id, path_id)
    mastery_map = {r["skill_label"]: r["mastery_score"] for r in mastery_records}

    unmet = []
    details = []

    for prereq in prerequisites:
        score = mastery_map.get(prereq, 0.0)
        status = _classify_mastery(score) if prereq in mastery_map else "not_assessed"
        details.append({
            "skill": prereq,
            "score": score,
            "status": status,
        })

        if score < config.MASTERY_THRESHOLD_REVIEW:
            unmet.append(prereq)

    is_ready = len(unmet) == 0

    return ReadinessResult(
        session_number=session_number,
        is_ready=is_ready,
        prerequisites=prerequisites,
        unmet_prerequisites=unmet,
        details=details,
    )


def get_mastery_summary(user_id: int, path_id: int) -> MasterySummary:
    """Aggregate mastery stats across all skills in a path."""
    records = get_skill_mastery(user_id, path_id)

    mastered = sum(1 for r in records if r.status == "mastered")
    review = sum(1 for r in records if r.status == "review")
    weak = sum(1 for r in records if r.status == "weak")
    not_assessed = sum(1 for r in records if r.status == "not_assessed")
    total = len(records)

    overall = (
        sum(r.mastery_score for r in records) / total
        if total > 0
        else 0.0
    )

    return MasterySummary(
        total_skills=total,
        mastered_count=mastered,
        review_count=review,
        weak_count=weak,
        not_assessed_count=not_assessed,
        overall_percentage=overall,
        skills=records,
    )
