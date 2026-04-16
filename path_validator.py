"""
path_validator.py – Path Validator & Corrector
-----------------------------------------------
LLM-powered ingestion gate (Step 1 of the learning pipeline).

Reads a learning path JSON file, extracts the career goal from the
filename, validates every session against that goal, and auto-corrects
any mismatched sessions.  Saves the corrected path and a human-readable
change log — never overwrites the original.

Uses the base Gemma model for generation; falls back to Gemini if the
local model's output can't be parsed after retries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
import llm_service

logger = logging.getLogger(__name__)


# ── data structures ──────────────────────────────────────────────────

@dataclass
class ValidationResult:
    session_number: int
    original_title: str
    relevance_score: int           # 0-100
    is_relevant: bool              # True if score >= 70
    issues: list[str]              # flagged problems
    explanation: str


@dataclass
class CorrectionResult:
    session_number: int
    original: dict                 # original session dict
    corrected: dict                # corrected session dict
    changes: list[str]             # human-readable list of changes


@dataclass
class PathValidationReport:
    career_goal: str
    filename: str
    total_sessions: int
    flagged_count: int
    corrected_count: int
    validations: list[ValidationResult] = field(default_factory=list)
    corrections: list[CorrectionResult] = field(default_factory=list)


# ── loading ──────────────────────────────────────────────────────────

def load_learning_path(filepath: str | Path) -> dict:
    """Parse a learning path JSON and extract the career goal from filename.

    Returns
    -------
    dict with keys: ``career_goal``, ``filename``, ``path_data`` (the full JSON).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Learning path not found: {filepath}")

    career_goal = filepath.stem  # e.g. "Game Dev" from "Game Dev.json"

    with open(filepath, "r", encoding="utf-8") as f:
        path_data = json.load(f)

    logger.info("Loaded learning path '%s' (%d sessions) for goal: %s",
                filepath.name, len(path_data.get("learning_path", [])), career_goal)

    return {
        "career_goal": career_goal,
        "filename": filepath.name,
        "filepath": str(filepath),
        "path_data": path_data,
    }


# ── validation ───────────────────────────────────────────────────────

_VALIDATE_SESSION_PROMPT = """\
You are an expert career-path reviewer.

Career Goal: {career_goal}

A learning path for this career goal contains the following session:

Session {session_number} of {total_sessions}
Title: {title}
Skills: {skills}
Objectives: {objectives}
Prerequisites: {prerequisites}

### TASK
Evaluate whether this session genuinely serves the career goal "{career_goal}".
Check for:
1. Skills that belong to a DIFFERENT career (e.g. gambling skills in a Game Dev path)
2. Objectives that don't align with the career goal
3. Content that seems auto-generated from a wrong occupation mapping

Return your assessment as JSON with exactly these keys:
{{"relevance_score": <int 0-100>, "is_relevant": <bool>, "issues": [<list of strings>], "explanation": "<one sentence>"}}

Rules:
* 100 = perfectly aligned with the career goal
* 0 = completely irrelevant or harmful to the path
* Score below 70 if skills/objectives clearly belong to a different career
* Be specific about what's wrong in the issues list
* ONLY output the JSON object. No extra text.

### YOUR JSON RESPONSE:
"""


def validate_session(session: dict, career_goal: str,
                     total_sessions: int) -> ValidationResult:
    """Use the LLM to check if a session aligns with the career goal."""
    prompt = _VALIDATE_SESSION_PROMPT.format(
        career_goal=career_goal,
        session_number=session.get("session_number", "?"),
        total_sessions=total_sessions,
        title=session.get("title", ""),
        skills=", ".join(session.get("skills", [])),
        objectives="\n".join(f"- {o}" for o in session.get("objectives", [])),
        prerequisites=", ".join(session.get("prerequisites", [])) or "None",
    )

    try:
        raw = llm_service.call_llm(prompt, task="generation")
        data = llm_service.parse_json_response(raw)
        return ValidationResult(
            session_number=session.get("session_number", 0),
            original_title=session.get("title", ""),
            relevance_score=int(data.get("relevance_score", 0)),
            is_relevant=bool(data.get("is_relevant", False)),
            issues=data.get("issues", []),
            explanation=str(data.get("explanation", "")),
        )
    except Exception as exc:
        logger.warning("Validation failed for session %s: %s",
                       session.get("session_number"), exc)
        # On parse failure, flag for manual review
        return ValidationResult(
            session_number=session.get("session_number", 0),
            original_title=session.get("title", ""),
            relevance_score=0,
            is_relevant=False,
            issues=[f"Validation failed: {exc}"],
            explanation="Could not validate — flagged for correction.",
        )


# ── correction ───────────────────────────────────────────────────────

_CORRECT_SESSION_PROMPT = """\
You are an expert curriculum designer.

Career Goal: {career_goal}

The following session in a {career_goal} learning path has been flagged as
misaligned. It contains skills or content that don't belong.

Session {session_number} of {total_sessions}
Current Title: {title}
Current Skills: {skills}
Current Objectives:
{objectives}
Prerequisites: {prerequisites}
Difficulty: {difficulty}
Duration: {duration} hours
Issues Found: {issues}

### TASK
Rewrite this session to genuinely serve the career goal "{career_goal}".
Keep the session position ({session_number} of {total_sessions}) and
difficulty level ({difficulty}) in mind — the content should be appropriate
for where this session sits in the learning sequence.

Return a JSON object with exactly these keys:
{{
  "title": "<new title>",
  "skills": [<list of 1-3 relevant skills>],
  "objectives": [<list of 2-4 learning objectives>],
  "comprehensive_guides": {{<skill_name>: "<brief guide content>"}},
  "learning_resources": {{
    "online_courses": [<1-3 resource names>],
    "tutorials": [<1-3 resource names>],
    "documentation": [<1-2 resource names>],
    "practice_platforms": [<1-2 platform names>],
    "communities": [<1-2 community names>]
  }}
}}

Rules:
* Skills must be directly relevant to "{career_goal}"
* Maintain the same difficulty level: {difficulty}
* Respect prerequisites: {prerequisites}
* Generate realistic resource names (real platforms and courses)
* ONLY output the JSON object. No extra text.

### YOUR JSON RESPONSE:
"""


def correct_session(session: dict, career_goal: str, total_sessions: int,
                    issues: list[str]) -> CorrectionResult:
    """Use the LLM to rewrite a misaligned session."""
    prompt = _CORRECT_SESSION_PROMPT.format(
        career_goal=career_goal,
        session_number=session.get("session_number", "?"),
        total_sessions=total_sessions,
        title=session.get("title", ""),
        skills=", ".join(session.get("skills", [])),
        objectives="\n".join(f"- {o}" for o in session.get("objectives", [])),
        prerequisites=", ".join(session.get("prerequisites", [])) or "None",
        difficulty=session.get("difficulty_level", "beginner"),
        duration=session.get("estimated_duration_hours", 8),
        issues="; ".join(issues),
    )

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt == 1:
                raw = llm_service.call_llm(prompt, task="generation",
                                           fallback_to_gemini=False)
            else:
                # Second attempt: fall back to Gemini
                raw = llm_service.call_gemini(prompt)

            data = llm_service.parse_json_response(raw)

            # Build corrected session, preserving fields the LLM shouldn't change
            corrected = dict(session)
            corrected["title"] = data.get("title", session.get("title", ""))
            corrected["skills"] = data.get("skills", session.get("skills", []))
            corrected["objectives"] = data.get("objectives", session.get("objectives", []))

            if "comprehensive_guides" in data:
                corrected["comprehensive_guides"] = data["comprehensive_guides"]
            if "learning_resources" in data:
                corrected["learning_resources"] = data["learning_resources"]

            # Update skill_details to match new skills
            corrected["skill_details"] = [
                {
                    "label": skill,
                    "description": f"Corrected skill for {career_goal}",
                    "difficulty": session.get("difficulty_level", "beginner"),
                    "priority": i + 1,
                    "relation_type": "essential",
                    "relevance_score": 0.8,
                }
                for i, skill in enumerate(corrected["skills"])
            ]

            # Compute changes
            changes = _compute_changes(session, corrected)

            return CorrectionResult(
                session_number=session.get("session_number", 0),
                original=session,
                corrected=corrected,
                changes=changes,
            )

        except Exception as exc:
            logger.warning("Correction attempt %d failed for session %s: %s",
                           attempt, session.get("session_number"), exc)
            if attempt == max_attempts:
                # Return original with a warning note
                return CorrectionResult(
                    session_number=session.get("session_number", 0),
                    original=session,
                    corrected=session,
                    changes=[f"CORRECTION FAILED: {exc} — session kept as-is"],
                )


def _compute_changes(original: dict, corrected: dict) -> list[str]:
    """Generate a human-readable list of what changed."""
    changes = []
    if original.get("title") != corrected.get("title"):
        changes.append(
            f"Title: \"{original.get('title')}\" -> \"{corrected.get('title')}\""
        )
    if original.get("skills") != corrected.get("skills"):
        changes.append(
            f"Skills: {original.get('skills')} -> {corrected.get('skills')}"
        )
    if original.get("objectives") != corrected.get("objectives"):
        changes.append("Objectives: rewritten")
    if original.get("learning_resources") != corrected.get("learning_resources"):
        changes.append("Resources: updated")
    if original.get("comprehensive_guides") != corrected.get("comprehensive_guides"):
        changes.append("Guides: rewritten")
    return changes


# ── orchestrator ─────────────────────────────────────────────────────

def validate_and_correct_path(
    filepath: str | Path,
) -> tuple[dict, str, PathValidationReport]:
    """Validate and correct an entire learning path.

    Returns
    -------
    (corrected_path_data, change_log, report)
        - corrected_path_data: the full corrected JSON dict
        - change_log: human-readable string of all changes
        - report: structured PathValidationReport
    """
    loaded = load_learning_path(filepath)
    career_goal = loaded["career_goal"]
    filename = loaded["filename"]
    path_data = loaded["path_data"]
    sessions = path_data.get("learning_path", [])
    total_sessions = len(sessions)

    report = PathValidationReport(
        career_goal=career_goal,
        filename=filename,
        total_sessions=total_sessions,
        flagged_count=0,
        corrected_count=0,
    )

    corrected_sessions = []
    log_lines = [
        f"Path Validation Report: {filename}",
        f"Career Goal: {career_goal}",
        f"Total Sessions: {total_sessions}",
        "=" * 60,
        "",
    ]

    for session in sessions:
        sn = session.get("session_number", "?")
        logger.info("Validating session %s: %s", sn, session.get("title", ""))
        log_lines.append(f"Session {sn}: {session.get('title', '')}")

        # Validate
        validation = validate_session(session, career_goal, total_sessions)
        report.validations.append(validation)

        log_lines.append(f"  Relevance Score: {validation.relevance_score}/100")

        if validation.is_relevant and validation.relevance_score >= 70:
            # Session is fine — keep as-is
            corrected_sessions.append(session)
            log_lines.append("  Status: PASS")
            log_lines.append("")
        else:
            # Session needs correction
            report.flagged_count += 1
            log_lines.append(f"  Status: FLAGGED")
            log_lines.append(f"  Issues: {'; '.join(validation.issues)}")

            correction = correct_session(
                session, career_goal, total_sessions, validation.issues
            )
            report.corrections.append(correction)

            if correction.changes and correction.changes[0].startswith("CORRECTION FAILED"):
                corrected_sessions.append(session)
                log_lines.append(f"  Correction: FAILED — kept original")
            else:
                report.corrected_count += 1
                corrected_sessions.append(correction.corrected)
                for change in correction.changes:
                    log_lines.append(f"  Changed: {change}")
                log_lines.append(f"  New Title: {correction.corrected.get('title', '')}")

            log_lines.append("")

    # Build corrected path
    corrected_path_data = dict(path_data)
    corrected_path_data["learning_path"] = corrected_sessions

    # Summary
    log_lines.append("=" * 60)
    log_lines.append(
        f"Summary: {report.flagged_count} flagged, "
        f"{report.corrected_count} corrected, "
        f"{total_sessions - report.flagged_count} passed"
    )

    change_log = "\n".join(log_lines)

    # Save corrected file
    config.CORRECTED_PATH_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    corrected_file = config.CORRECTED_PATH_DIR / f"{stem}_corrected.json"
    with open(corrected_file, "w", encoding="utf-8") as f:
        json.dump(corrected_path_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved corrected path to %s", corrected_file)

    # Save change log
    log_file = config.CORRECTED_PATH_DIR / f"{stem}_changes.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(change_log)
    logger.info("Saved change log to %s", log_file)

    return corrected_path_data, change_log, report
