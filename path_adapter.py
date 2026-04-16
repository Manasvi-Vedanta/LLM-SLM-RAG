"""
path_adapter.py – Adaptive learning path generator
----------------------------------------------------
Uses mastery scores to adapt the remaining learning path:
  - Score > 85%  → skip or compress session
  - Score 50-85% → flag for review, add practice
  - Score < 50%  → inject remediation session

Outputs a revised path as structured JSON + human-readable log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import config
import llm_service
from mastery_tracker import get_skill_mastery, SkillMastery
from skill_graph import build_skill_graph, get_esco_mapping

logger = logging.getLogger(__name__)


def _generate_remediation_content(
    skill_label: str,
    career_goal: str,
    difficulty: str,
) -> dict:
    """Use the LLM to generate focused remediation session content."""
    prompt = f"""\
You are a curriculum designer. A student pursuing "{career_goal}" is
struggling with "{skill_label}" (difficulty: {difficulty}).

Generate a focused remediation session as JSON:
{{
  "title": "Remediation: <concise title>",
  "skills": ["{skill_label}"],
  "objectives": ["<2-3 focused review objectives>"],
  "comprehensive_guides": {{"{skill_label}": "<300-500 word focused review guide>"}},
  "learning_resources": {{
    "online_courses": ["<1-2 beginner-friendly resources>"],
    "tutorials": ["<1-2 step-by-step tutorials>"],
    "documentation": [],
    "practice_platforms": ["<1 practice platform>"],
    "communities": []
  }}
}}

Focus on fundamentals and common misconceptions.
ONLY output the JSON object:
"""
    try:
        raw = llm_service.call_llm(prompt, task="generation")
        return llm_service.parse_json_response(raw)
    except Exception as exc:
        logger.warning("Remediation generation failed for %s: %s", skill_label, exc)
        return {
            "title": f"Remediation: {skill_label} Review",
            "skills": [skill_label],
            "objectives": [f"Review fundamentals of {skill_label}"],
            "comprehensive_guides": {skill_label: f"Review the basics of {skill_label}."},
            "learning_resources": {
                "online_courses": [], "tutorials": [], "documentation": [],
                "practice_platforms": [], "communities": [],
            },
        }


def _suggest_additional_resources(skill_label: str, career_goal: str) -> dict:
    """Use the LLM to suggest extra resources for weak skills."""
    prompt = f"""\
Suggest 3 high-quality learning resources for "{skill_label}" aimed at
someone pursuing a career as "{career_goal}" who is struggling with this topic.

Return JSON:
{{"online_courses": ["<name>"], "tutorials": ["<name>"], "practice_platforms": ["<name>"]}}

Focus on beginner-friendly, free resources. ONLY output JSON:
"""
    try:
        raw = llm_service.call_llm(prompt, task="generation")
        return llm_service.parse_json_response(raw)
    except Exception:
        return {"online_courses": [], "tutorials": [], "practice_platforms": []}


def adapt_path(
    user_id: int,
    path_id: int,
    corrected_path: dict,
    career_goal: str = "",
) -> tuple[dict, str]:
    """Adapt a learning path based on mastery scores.

    Parameters
    ----------
    user_id : int
        Database user ID.
    path_id : int
        Database learning path ID.
    corrected_path : dict
        The corrected learning path JSON.
    career_goal : str
        Career goal label.

    Returns
    -------
    (adapted_path, adaptation_log)
        - adapted_path: the full adapted JSON dict
        - adaptation_log: human-readable string of all changes
    """
    sessions = corrected_path.get("learning_path", [])
    mastery_records = get_skill_mastery(user_id, path_id)
    mastery_map = {r.skill_label: r for r in mastery_records}

    # Build prerequisite DAG and validate the original ordering
    graph = build_skill_graph(sessions)
    ordering_violations = graph.validate_ordering(sessions)
    esco_map = get_esco_mapping(sessions)

    adapted_sessions = []
    log_lines = [
        f"Path Adaptation Report",
        f"Career Goal: {career_goal}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Skills assessed: {len(mastery_records)}",
        f"Skill graph: {len(graph.nodes)} nodes, "
        f"{sum(len(v) for v in graph.edges.values())} edges",
        f"Ordering violations in original path: {len(ordering_violations)}",
        "=" * 60,
        "",
    ]
    if ordering_violations:
        log_lines.append("Prerequisite violations detected:")
        for v in ordering_violations[:10]:
            log_lines.append(f"  - {v}")
        log_lines.append("")

    remediation_queue = []  # (insert_before_session_number, remediation_session)

    for session in sessions:
        sn = session.get("session_number", 0)
        title = session.get("title", "")
        skills = session.get("skills", [])
        difficulty = session.get("difficulty_level", "beginner")

        log_lines.append(f"Session {sn}: {title}")

        # Check mastery for all skills in this session (using decayed scores)
        skill_scores = []
        for skill in skills:
            mastery = mastery_map.get(skill)
            if mastery:
                skill_scores.append((skill, mastery.decayed_score, mastery.status))
            else:
                skill_scores.append((skill, -1, "not_assessed"))

        # Determine session action
        assessed_scores = [(s, sc, st) for s, sc, st in skill_scores if sc >= 0]

        if not assessed_scores:
            # Not yet assessed — keep as-is
            adapted_sessions.append(session)
            log_lines.append("  Action: KEEP (not yet assessed)")
            log_lines.append("")
            continue

        avg_score = sum(sc for _, sc, _ in assessed_scores) / len(assessed_scores)

        if avg_score >= config.MASTERY_THRESHOLD_SKIP:
            # All skills mastered — compress
            compressed = dict(session)
            compressed["status"] = "compressed"
            compressed["original_duration"] = session.get("estimated_duration_hours", 8)
            compressed["estimated_duration_hours"] = max(1, session.get("estimated_duration_hours", 8) // 4)
            compressed["objectives"] = [f"Quick review: {title}"]
            adapted_sessions.append(compressed)
            log_lines.append(f"  Avg score: {avg_score:.1f}% -> COMPRESSED (review only)")
            log_lines.append(f"  Duration: {compressed['original_duration']}h -> {compressed['estimated_duration_hours']}h")
            log_lines.append("")

        elif avg_score >= config.MASTERY_THRESHOLD_REVIEW:
            # Needs review — keep with review flag
            review_session = dict(session)
            review_session["status"] = "review"

            # Identify weak skills within the session
            weak_skills = [s for s, sc, _ in assessed_scores if sc < config.MASTERY_THRESHOLD_SKIP]
            if weak_skills:
                review_session["review_focus"] = weak_skills
                extra = _suggest_additional_resources(weak_skills[0], career_goal)
                existing_resources = review_session.get("learning_resources", {})
                for key, vals in extra.items():
                    if key in existing_resources:
                        existing_resources[key] = list(set(existing_resources[key] + vals))
                review_session["learning_resources"] = existing_resources

            adapted_sessions.append(review_session)
            log_lines.append(f"  Avg score: {avg_score:.1f}% -> REVIEW")
            if weak_skills:
                log_lines.append(f"  Focus on: {', '.join(weak_skills)}")
                log_lines.append(f"  Additional resources suggested")
            log_lines.append("")

        else:
            # Weak — inject remediation before this session
            weak_skills = [s for s, sc, _ in assessed_scores if sc < config.MASTERY_THRESHOLD_REVIEW]

            for skill in weak_skills:
                remediation = _generate_remediation_content(skill, career_goal, difficulty)
                remediation["session_number"] = 0  # will be renumbered
                remediation["difficulty_level"] = "beginner"
                remediation["estimated_duration_hours"] = 4
                remediation["prerequisites"] = []
                remediation["skill_details"] = [{
                    "label": skill,
                    "description": f"Remediation for {skill}",
                    "difficulty": "beginner",
                    "priority": 1,
                    "relation_type": "essential",
                    "relevance_score": 1.0,
                }]
                remediation["status"] = "remediation"
                remediation_queue.append((sn, remediation))
                log_lines.append(f"  Skill '{skill}': {[sc for s, sc, _ in assessed_scores if s == skill][0]:.1f}% -> REMEDIATION INJECTED")

            # Keep original session
            original = dict(session)
            original["status"] = "pending_review"
            adapted_sessions.append(original)
            log_lines.append(f"  Avg score: {avg_score:.1f}% -> REMEDIATION + REVIEW")
            log_lines.append("")

    # Insert remediation sessions at correct positions
    for insert_before, remediation in sorted(remediation_queue, key=lambda x: x[0], reverse=True):
        # Find the index of the session to insert before
        for i, s in enumerate(adapted_sessions):
            if s.get("session_number") == insert_before:
                adapted_sessions.insert(i, remediation)
                break

    # Graph-constrained reorder: fix any prerequisite violations introduced
    # by remediation injection or present in the original path.
    reorder_graph = build_skill_graph(adapted_sessions)
    pre_violation_count = len(reorder_graph.validate_ordering(adapted_sessions))
    if pre_violation_count > 0:
        adapted_sessions = reorder_graph.constrained_reorder(adapted_sessions)
        post_violation_count = len(
            build_skill_graph(adapted_sessions).validate_ordering(adapted_sessions)
        )
        log_lines.append(
            f"Graph-constrained reorder: {pre_violation_count} -> "
            f"{post_violation_count} prerequisite violations"
        )
        log_lines.append("")

    # Renumber all sessions
    for i, session in enumerate(adapted_sessions, 1):
        session["session_number"] = i

    # Build adapted path
    adapted_path = dict(corrected_path)
    adapted_path["learning_path"] = adapted_sessions
    needs_review = [
        m.skill_label for m in mastery_records if m.needs_review
    ]
    adapted_path["adaptation_metadata"] = {
        "adapted_at": datetime.now().isoformat(),
        "original_session_count": len(sessions),
        "adapted_session_count": len(adapted_sessions),
        "mastery_records_used": len(mastery_records),
        "skill_graph_nodes": len(graph.nodes),
        "skill_graph_edges": sum(len(v) for v in graph.edges.values()),
        "original_ordering_violations": len(ordering_violations),
        "skills_needing_spaced_review": needs_review,
        "esco_uri_map": esco_map,
    }

    # Summary
    statuses = [s.get("status", "normal") for s in adapted_sessions]
    log_lines.append("=" * 60)
    log_lines.append(f"Summary:")
    log_lines.append(f"  Original sessions: {len(sessions)}")
    log_lines.append(f"  Adapted sessions: {len(adapted_sessions)}")
    log_lines.append(f"  Compressed: {statuses.count('compressed')}")
    log_lines.append(f"  Review: {statuses.count('review')}")
    log_lines.append(f"  Remediation added: {statuses.count('remediation')}")
    log_lines.append(f"  Unchanged: {statuses.count('normal') + statuses.count('pending_review')}")

    adaptation_log = "\n".join(log_lines)

    # Save to disk
    config.CORRECTED_PATH_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = corrected_path.get("_filename", "path")
    adapted_file = config.CORRECTED_PATH_DIR / f"{filename}_adapted_{timestamp}.json"
    with open(adapted_file, "w", encoding="utf-8") as f:
        json.dump(adapted_path, f, indent=2, ensure_ascii=False)
    logger.info("Saved adapted path to %s", adapted_file)

    log_file = config.CORRECTED_PATH_DIR / f"{filename}_adaptation_{timestamp}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(adaptation_log)

    return adapted_path, adaptation_log
