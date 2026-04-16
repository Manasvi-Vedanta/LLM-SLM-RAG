"""
mastery_tracker.py – Skill mastery tracking with Ebbinghaus time-decay
-----------------------------------------------------------------------
Aggregates per-skill mastery scores, applies a forgetting-curve decay
based on time since last assessment, checks prerequisite readiness, and
provides summaries for path adaptation decisions.

Decay formula:
    decayed = max(FLOOR, raw_score * exp(-LAMBDA * days_since_assessed))

Skills that have not been re-assessed in a long time have a "decayed"
score that drops below the raw score, signalling a need for review.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config
import database

logger = logging.getLogger(__name__)


# ── data structures ──────────────────────────────────────────────────

@dataclass
class SkillMastery:
    skill_label: str
    session_number: int
    mastery_score: float          # 0-100 raw, as stored in DB
    decayed_score: float          # raw_score after Ebbinghaus decay
    attempts_count: int
    status: str                   # "mastered", "review", "weak", "not_assessed"
    last_assessed_at: str | None = None
    days_since_assessed: float = 0.0
    needs_review: bool = False    # True when decayed dropped below mastery threshold


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


# ── decay helpers ────────────────────────────────────────────────────

def _apply_decay(raw_score: float, last_assessed_at: str | None) -> tuple[float, float]:
    """Apply Ebbinghaus exponential decay to a raw mastery score.

    Returns
    -------
    (decayed_score, days_since_assessed)
    """
    if not last_assessed_at or raw_score <= 0:
        return raw_score, 0.0

    try:
        # SQLite CURRENT_TIMESTAMP format is "YYYY-MM-DD HH:MM:SS" (UTC, no tz)
        assessed_dt = datetime.strptime(last_assessed_at, "%Y-%m-%d %H:%M:%S")
        assessed_dt = assessed_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return raw_score, 0.0

    now = datetime.now(timezone.utc)
    days = max(0.0, (now - assessed_dt).total_seconds() / 86400.0)
    if days == 0:
        return raw_score, 0.0

    decayed = raw_score * math.exp(-config.MASTERY_DECAY_LAMBDA * days)
    decayed = max(config.MASTERY_DECAY_FLOOR, decayed)
    return decayed, days


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
    """Get all mastery records for a learning path with decayed scores."""
    rows = database.get_skill_mastery(user_id, path_id)
    result = []
    for r in rows:
        raw = r["mastery_score"]
        last = r.get("last_assessed_at") if isinstance(r, dict) else r["last_assessed_at"]
        decayed, days = _apply_decay(raw, last)
        needs_review = (
            raw >= config.MASTERY_THRESHOLD_SKIP
            and decayed < config.MASTERY_THRESHOLD_SKIP
        )
        result.append(
            SkillMastery(
                skill_label=r["skill_label"],
                session_number=r["session_number"],
                mastery_score=raw,
                decayed_score=decayed,
                attempts_count=r["attempts_count"],
                status=_classify_mastery(decayed),
                last_assessed_at=last,
                days_since_assessed=days,
                needs_review=needs_review,
            )
        )
    return result


def get_session_readiness(
    user_id: int,
    path_id: int,
    session_number: int,
    corrected_path: dict,
) -> ReadinessResult:
    """Check if prerequisite skills are mastered for a session.

    Uses decayed mastery scores so a long-dormant prereq is flagged
    even if its raw score is high.
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

    # Use decayed scores from get_skill_mastery
    mastery_records = get_skill_mastery(user_id, path_id)
    mastery_map = {m.skill_label: m for m in mastery_records}

    unmet = []
    details = []

    for prereq in prerequisites:
        m = mastery_map.get(prereq)
        if m is None:
            score = 0.0
            status = "not_assessed"
        else:
            score = m.decayed_score
            status = m.status
        details.append({
            "skill": prereq,
            "score": round(score, 1),
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
    """Aggregate mastery stats across all skills in a path (decayed)."""
    records = get_skill_mastery(user_id, path_id)

    mastered = sum(1 for r in records if r.status == "mastered")
    review = sum(1 for r in records if r.status == "review")
    weak = sum(1 for r in records if r.status == "weak")
    not_assessed = sum(1 for r in records if r.status == "not_assessed")
    total = len(records)

    overall = (
        sum(r.decayed_score for r in records) / total
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
