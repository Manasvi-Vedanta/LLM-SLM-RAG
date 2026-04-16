"""
ingestion.py
------------
Responsible for:
  1. Loading PDF documents from the Dataset folder.
  2. Loading raw text content (guides, scraped web pages).
  3. Loading transcript files.
  4. Splitting them into overlapping text chunks.
  5. Returning a list of LangChain Document objects ready for embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def _discover_pdfs(directory: Path) -> List[Path]:
    """Return sorted list of PDF paths inside *directory*."""
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {directory}")
    logger.info("Discovered %d PDF(s) in %s", len(pdfs), directory)
    return pdfs


def load_pdfs(directory: Path | None = None) -> List[Document]:
    """
    Load every PDF in *directory* (defaults to ``config.DATASET_DIR``).

    Returns a flat list of LangChain ``Document`` objects – one per page –
    with metadata containing the source filename and page number.
    """
    directory = directory or config.DATASET_DIR
    all_docs: List[Document] = []

    for pdf_path in _discover_pdfs(directory):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for doc in pages:
            doc.metadata["source"] = pdf_path.name
        all_docs.extend(pages)
        logger.info("  -> Loaded %d pages from %s", len(pages), pdf_path.name)

    logger.info("Total pages loaded: %d", len(all_docs))
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split *documents* into smaller overlapping chunks.

    Each resulting ``Document`` inherits the original metadata (source, page)
    so we can always trace a chunk back to its origin.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked %d pages into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def ingest(directory: Path | None = None) -> List[Document]:
    """Convenience wrapper: load PDFs → chunk → return chunks."""
    docs = load_pdfs(directory)
    return chunk_documents(docs)


# ──────────────────────────────────────────────
# Text content loader (guides, scraped web pages)
# ──────────────────────────────────────────────

def load_text_content(text: str, metadata: dict | None = None) -> List[Document]:
    """Wrap a raw text string into LangChain Document(s).

    Parameters
    ----------
    text : str
        Raw text content (e.g. comprehensive_guides, scraped web page).
    metadata : dict, optional
        Metadata to attach (source_type, skill_label, etc.).

    Returns
    -------
    List[Document]
        A single-element list containing the Document.
        Returns empty list if text is blank.
    """
    text = (text or "").strip()
    if not text:
        return []

    metadata = metadata or {}
    doc = Document(page_content=text, metadata=metadata)
    logger.debug("Loaded text content (%d chars, source_type=%s)",
                 len(text), metadata.get("source_type", "unknown"))
    return [doc]


# ──────────────────────────────────────────────
# Transcript file loader
# ──────────────────────────────────────────────

def load_transcript(transcript_path: Path, metadata: dict | None = None) -> List[Document]:
    """Load a transcript .txt file into a LangChain Document.

    Parameters
    ----------
    transcript_path : Path
        Path to the transcript file.
    metadata : dict, optional
        Metadata to attach (source_type, video_title, url, etc.).

    Returns
    -------
    List[Document]
        A single-element list, or empty if the file is missing/empty.
    """
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        logger.warning("Transcript file not found: %s", transcript_path)
        return []

    text = transcript_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning("Empty transcript file: %s", transcript_path)
        return []

    metadata = metadata or {}
    metadata.setdefault("source_type", "transcript")
    metadata.setdefault("source", transcript_path.name)

    doc = Document(page_content=text, metadata=metadata)
    logger.info("Loaded transcript (%d chars) from %s", len(text), transcript_path.name)
    return [doc]
