"""
ingestion.py
------------
Responsible for:
  1. Loading PDF documents from the Dataset folder.
  2. Splitting them into overlapping text chunks.
  3. Returning a list of LangChain Document objects ready for embedding.
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
