"""
session_orchestrator.py – Session ingestion orchestrator
---------------------------------------------------------
Single entry point for making a session's content searchable.
Called lazily when a user starts studying a session.

Ties together: session_mapper → vector_store → database updates.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import json

import config
import database
from session_mapper import map_session_content, SessionContent
from vector_store import (
    build_session_vectorstore,
    session_vectorstore_exists,
)

logger = logging.getLogger(__name__)

# Cache of source details per session (populated during ingestion)
_session_materials: dict[str, dict] = {}


def session_id_for(path_id: str, session_number: int) -> str:
    """Canonical session ID used across all components."""
    return f"{path_id}_session_{session_number}"


def ingest_session(
    session: dict,
    path_id: str,
    career_goal: str = "",
    force_rebuild: bool = False,
) -> str:
    """Ingest a session's content and build its FAISS index.

    Parameters
    ----------
    session : dict
        A single session dict from the (corrected) learning path.
    path_id : str
        Identifier for the learning path (e.g. "ml_engineer" or a DB id).
    career_goal : str
        Career goal label for metadata context.
    force_rebuild : bool
        If True, rebuild even if the vectorstore already exists.

    Returns
    -------
    str
        The session_id (e.g. "ml_engineer_session_3").
    """
    session_number = session.get("session_number", 0)
    sid = session_id_for(path_id, session_number)

    # Skip if already built (unless forced)
    if not force_rebuild and session_vectorstore_exists(sid):
        logger.info("Session %s already ingested — skipping.", sid)
        return sid

    logger.info("Ingesting session %d for path '%s'...", session_number, path_id)

    # Collect and chunk all content
    content = map_session_content(session, path_id, career_goal)

    if not content.documents:
        logger.warning("Session %d: no content to index.", session_number)
        return sid

    # Build FAISS index
    build_session_vectorstore(sid, content.documents)

    # Update database
    try:
        database.update_session_progress(
            path_id=int(path_id) if path_id.isdigit() else 0,
            session_number=session_number,
            vectorstore_built=1,
        )
    except Exception as exc:
        # Non-fatal: the vectorstore is built even if the DB update fails
        logger.warning("Could not update session_progress for %s: %s", sid, exc)

    # Cache materials metadata for the materials endpoint
    _session_materials[sid] = {
        "content_sources": content.content_sources,
        "source_details": content.source_details,
        "total_chunks": len(content.documents),
    }

    logger.info(
        "Session %d ingested: %d chunks indexed (sources: %s)",
        session_number,
        len(content.documents),
        content.content_sources,
    )

    return sid


def get_session_materials(session_id: str) -> dict | None:
    """Return cached materials metadata for a session, or None if not ingested."""
    return _session_materials.get(session_id)


def ingest_all_sessions(
    path_data: dict,
    path_id: str,
    career_goal: str = "",
) -> list[str]:
    """Ingest all sessions in a learning path (non-lazy, for batch use).

    Returns list of session IDs.
    """
    sessions = path_data.get("learning_path", [])
    session_ids = []
    for session in sessions:
        sid = ingest_session(session, path_id, career_goal)
        session_ids.append(sid)
    return session_ids
