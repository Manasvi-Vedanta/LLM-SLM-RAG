"""
review_scheduler.py – Spaced repetition review scheduling
------------------------------------------------------------
Computes when each skill should next be reviewed, based on the
Ebbinghaus decay model used in mastery_tracker.

The review interval is the number of days until a skill's decayed
score would drop to MASTERY_THRESHOLD_SKIP. Reviewing a skill
resets last_assessed_at and the interval restarts from the new score.

For a skill with current mastery_score S (raw) and decay rate λ:
    decayed(d) = max(FLOOR, S * exp(-λ * d))
    set decayed(d) = THRESHOLD_SKIP
    →  d = ln(S / THRESHOLD_SKIP) / λ       when S > THRESHOLD_SKIP
    →  d = 0                                 when S <= THRESHOLD_SKIP

A skill stronger than the mastery threshold gets a longer interval.
A skill at or below threshold is due immediately.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import config
import database

logger = logging.getLogger(__name__)


@dataclass
class ReviewItem:
    """A skill scheduled for review."""
    skill_label: str
    session_number: int
    mastery_score: float          # raw, as stored
    decayed_score: float          # after time-decay
    last_assessed_at: str | None
    days_since_assessed: float
    interval_days: float          # recommended review interval
    next_review_at: str | None    # ISO timestamp
    days_until_review: float      # negative → overdue
    urgency: str                  # "overdue", "due_today", "due_this_week", "scheduled"


@dataclass
class ReviewSchedule:
    items: list[ReviewItem] = field(default_factory=list)
    overdue_count: int = 0
    due_today_count: int = 0
    due_this_week_count: int = 0
    scheduled_count: int = 0


def _compute_interval_days(mastery_score: float) -> float:
    """Days until decayed score would hit MASTERY_THRESHOLD_SKIP.

    Skills at or below the threshold return 0 (review now).
    """
    threshold = config.MASTERY_THRESHOLD_SKIP
    lam = config.MASTERY_DECAY_LAMBDA

    if mastery_score <= threshold or lam <= 0:
        return 0.0
    # Solve S * exp(-λ * d) = threshold  →  d = ln(S/threshold) / λ
    return math.log(mastery_score / threshold) / lam


def _classify_urgency(days_until_review: float) -> str:
    if days_until_review < 0:
        return "overdue"
    if days_until_review < 1:
        return "due_today"
    if days_until_review <= 7:
        return "due_this_week"
    return "scheduled"


def build_review_schedule(user_id: int, path_id: int) -> ReviewSchedule:
    """Build the spaced-repetition review queue for a user's skills."""
    rows = database.get_skill_mastery(user_id, path_id)
    now = datetime.now(timezone.utc)
    schedule = ReviewSchedule()

    for r in rows:
        raw_score = r["mastery_score"]
        last_assessed = r.get("last_assessed_at")

        if last_assessed:
            try:
                assessed_dt = datetime.strptime(last_assessed, "%Y-%m-%d %H:%M:%S")
                assessed_dt = assessed_dt.replace(tzinfo=timezone.utc)
                days_elapsed = max(0.0, (now - assessed_dt).total_seconds() / 86400.0)
            except (ValueError, TypeError):
                assessed_dt = None
                days_elapsed = 0.0
        else:
            assessed_dt = None
            days_elapsed = 0.0

        # Decayed score (mirrors mastery_tracker logic)
        if days_elapsed > 0 and raw_score > 0:
            decayed = max(
                config.MASTERY_DECAY_FLOOR,
                raw_score * math.exp(-config.MASTERY_DECAY_LAMBDA * days_elapsed),
            )
        else:
            decayed = raw_score

        interval = _compute_interval_days(raw_score)

        if assessed_dt:
            next_review_dt = assessed_dt + timedelta(days=interval)
            next_review_iso = next_review_dt.isoformat()
            days_until = (next_review_dt - now).total_seconds() / 86400.0
        else:
            next_review_iso = None
            days_until = 0.0  # never assessed → review-worthy

        urgency = _classify_urgency(days_until)

        item = ReviewItem(
            skill_label=r["skill_label"],
            session_number=r["session_number"],
            mastery_score=round(raw_score, 1),
            decayed_score=round(decayed, 1),
            last_assessed_at=last_assessed,
            days_since_assessed=round(days_elapsed, 2),
            interval_days=round(interval, 2),
            next_review_at=next_review_iso,
            days_until_review=round(days_until, 2),
            urgency=urgency,
        )
        schedule.items.append(item)

        if urgency == "overdue":
            schedule.overdue_count += 1
        elif urgency == "due_today":
            schedule.due_today_count += 1
        elif urgency == "due_this_week":
            schedule.due_this_week_count += 1
        else:
            schedule.scheduled_count += 1

    # Sort: most overdue first
    schedule.items.sort(key=lambda it: it.days_until_review)

    logger.info(
        "Review schedule: %d overdue, %d today, %d this week, %d scheduled",
        schedule.overdue_count, schedule.due_today_count,
        schedule.due_this_week_count, schedule.scheduled_count,
    )

    return schedule
