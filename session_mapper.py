"""
session_mapper.py – Session-Document Mapper
---------------------------------------------
Maps each learning path session to its ingestible content.

For each session, collects documents in priority order:
  1. User-provided PDFs (highest priority)
  2. Scraped web content (resolved from resource names)
  3. comprehensive_guides text from the JSON (always available)

Optionally extracts video transcripts if dependencies are installed.
All content is chunked and returned as LangChain Documents ready for
FAISS indexing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_core.documents import Document

import config
from ingestion import load_pdfs, load_text_content, load_transcript, chunk_documents
import web_resource_resolver
import transcript_extractor

logger = logging.getLogger(__name__)


@dataclass
class SessionContent:
    """Content collected for a single session."""
    session_id: str
    session_number: int
    skill_labels: list[str]
    documents: List[Document]         # chunked, ready for embedding
    content_sources: dict = field(default_factory=dict)
    # content_sources tracks counts: {"pdf": 2, "web": 3, "guide": 1, "transcript": 0}
    source_details: list[dict] = field(default_factory=list)
    # source_details: [{"type": "pdf", "name": "file.pdf", "path": "...", "url": "..."}, ...]


def _session_id(path_id: str, session_number: int) -> str:
    """Generate a unique session ID."""
    return f"{path_id}_session_{session_number}"


def _get_session_pdf_dir(path_id: str, session_number: int) -> Path:
    """Path where user can place PDFs for a specific session."""
    return config.SESSION_CONTENT_DIR / path_id / f"session_{session_number}"


def map_session_content(
    session: dict,
    path_id: str,
    career_goal: str = "",
) -> SessionContent:
    """Collect and chunk all ingestible content for a session.

    Parameters
    ----------
    session : dict
        A single session dict from the learning path JSON.
    path_id : str
        Identifier for the learning path (e.g. "ml_engineer").
    career_goal : str
        Career goal for metadata context.

    Returns
    -------
    SessionContent
        Chunked documents ready for FAISS indexing.
    """
    session_number = session.get("session_number", 0)
    sid = _session_id(path_id, session_number)
    skills = session.get("skills", [])
    primary_skill = skills[0] if skills else session.get("title", "unknown")

    all_docs: List[Document] = []
    sources = {"pdf": 0, "web": 0, "guide": 0, "transcript": 0}
    details: list[dict] = []

    base_metadata = {
        "session_id": sid,
        "session_number": session_number,
        "career_goal": career_goal,
        "skill_labels": ", ".join(skills),
    }

    # ── 1. User-provided PDFs (highest priority) ──
    pdf_dir = _get_session_pdf_dir(path_id, session_number)
    if pdf_dir.exists():
        try:
            pdf_docs = load_pdfs(pdf_dir)
            seen_pdfs: set[str] = set()
            for doc in pdf_docs:
                doc.metadata.update(base_metadata)
                doc.metadata["source_type"] = "pdf"
                fname = doc.metadata.get("source", "")
                if fname and fname not in seen_pdfs:
                    seen_pdfs.add(fname)
                    details.append({
                        "type": "pdf",
                        "name": fname,
                        "path": str(pdf_dir / fname),
                    })
            all_docs.extend(pdf_docs)
            sources["pdf"] = len(pdf_docs)
            logger.info("Session %d: loaded %d PDF pages", session_number, len(pdf_docs))
        except FileNotFoundError:
            pass  # no PDFs in the directory

    # ── 2. Scraped web content + video URL detection ──
    resources = session.get("learning_resources", {})
    has_resources = any(resources.get(k) for k in resources)
    web_docs: List[Document] = []
    video_resources: list[dict] = []
    if has_resources:
        web_docs, video_resources = web_resource_resolver.resolve_and_scrape_session_resources(
            resources, primary_skill
        )
        for doc in web_docs:
            doc.metadata.update(base_metadata)
            details.append({
                "type": "web",
                "name": doc.metadata.get("resource_name", ""),
                "url": doc.metadata.get("url", ""),
                "category": doc.metadata.get("resource_category", ""),
            })
        all_docs.extend(web_docs)
        sources["web"] = len(web_docs)

    # ── 3. comprehensive_guides text (always available) ──
    guides = session.get("comprehensive_guides", {})
    for skill_name, guide_text in guides.items():
        guide_docs = load_text_content(guide_text, metadata={
            **base_metadata,
            "source_type": "guide",
            "source": f"guide:{skill_name}",
            "skill_label": skill_name,
        })
        all_docs.extend(guide_docs)
        details.append({
            "type": "guide",
            "name": skill_name,
            "chars": len(guide_text),
            "content": guide_text.strip(),
        })
    sources["guide"] = len(guides)

    # ── 4. Video transcripts ──
    # Collect video URLs from: resolver-detected + raw resource names that are URLs
    video_urls: list[dict] = list(video_resources)  # already {"name", "url", "category"}
    seen_video_urls = {v["url"] for v in video_urls}

    for category in ["online_courses", "tutorials"]:
        for item in resources.get(category, []):
            if isinstance(item, str) and transcript_extractor.is_video_url(item):
                if item not in seen_video_urls:
                    video_urls.append({"name": item, "url": item, "category": category})
                    seen_video_urls.add(item)

    if video_urls:
        if transcript_extractor.is_available():
            transcript_dir = config.SESSION_CONTENT_DIR / path_id / f"session_{session_number}" / "transcripts"
            urls_to_extract = [v["url"] for v in video_urls]
            transcript_paths = transcript_extractor.extract_transcripts_batch(
                urls_to_extract, transcript_dir
            )
            extracted_set: set[str] = set()
            for i, tp in enumerate(transcript_paths):
                t_docs = load_transcript(tp, metadata={
                    **base_metadata,
                    "source_type": "transcript",
                })
                all_docs.extend(t_docs)
                url = urls_to_extract[i] if i < len(urls_to_extract) else ""
                extracted_set.add(url)
                # Find the matching name from video_urls
                vname = next((v["name"] for v in video_urls if v["url"] == url), tp.name)
                details.append({
                    "type": "transcript",
                    "name": vname,
                    "url": url,
                    "path": str(tp),
                    "status": "extracted",
                })
            sources["transcript"] = len(transcript_paths)

            # Any video URLs that failed extraction — still show in materials
            for v in video_urls:
                if v["url"] not in extracted_set:
                    details.append({
                        "type": "transcript",
                        "name": v["name"],
                        "url": v["url"],
                        "status": "extraction_failed",
                    })
        else:
            # yt-dlp/whisper not installed — still show video resources in materials
            logger.info(
                "Transcript extraction unavailable — %d video resource(s) will be "
                "shown as direct links in materials.", len(video_urls)
            )
            for v in video_urls:
                details.append({
                    "type": "transcript",
                    "name": v["name"],
                    "url": v["url"],
                    "status": "unavailable",
                })
            sources["transcript"] = 0

    # ── Chunk all collected documents ──
    if not all_docs:
        logger.warning("Session %d: no content collected", session_number)
        return SessionContent(
            session_id=sid,
            session_number=session_number,
            skill_labels=skills,
            documents=[],
            content_sources=sources,
            source_details=details,
        )

    chunks = chunk_documents(all_docs)
    logger.info(
        "Session %d: %d raw docs -> %d chunks (pdf=%d, web=%d, guide=%d, transcript=%d)",
        session_number, len(all_docs), len(chunks),
        sources["pdf"], sources["web"], sources["guide"], sources["transcript"],
    )

    return SessionContent(
        session_id=sid,
        session_number=session_number,
        skill_labels=skills,
        documents=chunks,
        content_sources=sources,
        source_details=details,
    )
