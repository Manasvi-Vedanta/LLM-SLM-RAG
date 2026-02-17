"""
vector_store.py  –  The **Actor** (Retriever)
----------------------------------------------
Wraps a FAISS vector store with cosine-similarity search.

Public API
~~~~~~~~~~
* ``build_vectorstore(chunks)`` – create a new FAISS index from document chunks.
* ``load_vectorstore()``        – load a previously persisted index from disk.
* ``retrieve(query, k, threshold)`` – search and apply the scope gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config

logger = logging.getLogger(__name__)


# ── tiny value-object returned by retrieve() ─────────────────────────
@dataclass
class RetrievalResult:
    """Holds the chunks returned by the Actor along with their scores."""

    in_scope: bool
    """``False`` when the best chunk falls below the similarity threshold."""

    chunks: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    @property
    def best_chunk(self) -> Optional[Document]:
        return self.chunks[0] if self.chunks else None

    @property
    def best_score(self) -> Optional[float]:
        return self.scores[0] if self.scores else None


# ── embedding helper ──────────────────────────────────────────────────
def _get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},   # unit vectors → IP ≈ cosine
    )


# ── build / persist / load ────────────────────────────────────────────
def build_vectorstore(
    chunks: List[Document],
    persist_dir: Path | None = None,
    model_name: str | None = None,
) -> FAISS:
    """
    Create a FAISS index from *chunks*, persist it to disk, and return it.

    Uses **Inner Product** distance so that, with L2-normalised embeddings,
    the score equals cosine similarity  (higher = more similar).
    """
    persist_dir = persist_dir or config.VECTORSTORE_DIR
    embeddings = _get_embeddings(model_name)

    logger.info("Building FAISS index from %d chunks …", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embeddings)

    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir))
    logger.info("FAISS index saved to %s", persist_dir)
    return vectorstore


def load_vectorstore(
    persist_dir: Path | None = None,
    model_name: str | None = None,
) -> FAISS:
    """Load a previously persisted FAISS index."""
    persist_dir = persist_dir or config.VECTORSTORE_DIR
    embeddings = _get_embeddings(model_name)
    logger.info("Loading FAISS index from %s …", persist_dir)
    return FAISS.load_local(
        str(persist_dir), embeddings, allow_dangerous_deserialization=True
    )


# ── retrieval with scope gate ─────────────────────────────────────────
def retrieve(
    vectorstore: FAISS,
    query: str,
    k: int | None = None,
    threshold: float | None = None,
) -> RetrievalResult:
    """
    Retrieve top-*k* chunks for *query*.

    **Scope gate**: if the best chunk's similarity score is below
    *threshold*, the result is flagged ``in_scope=False``.
    """
    k = k or config.TOP_K
    threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD

    # BGE models perform better when queries (not documents) are prefixed
    # with an instruction string.  This is a no-op if the config value is empty.
    query_prefix = getattr(config, "EMBEDDING_QUERY_INSTRUCTION", "") or ""
    prefixed_query = query_prefix + query

    # similarity_search_with_score returns (doc, L2_distance) tuples.
    # Lower L2 distance = more similar.  For L2-normalised embeddings
    # the exact relationship is:  cosine_similarity = 1 - (L2² / 2).
    raw_results: List[Tuple[Document, float]] = (
        vectorstore.similarity_search_with_score(prefixed_query, k=k)
    )

    if not raw_results:
        return RetrievalResult(in_scope=False)

    # Convert L2 distance → cosine similarity and sort descending
    converted: List[Tuple[Document, float]] = []
    for doc, l2_dist in raw_results:
        cos_sim = 1.0 - (float(l2_dist) ** 2) / 2.0
        converted.append((doc, cos_sim))
    converted.sort(key=lambda x: x[1], reverse=True)

    chunks = [doc for doc, _ in converted]
    scores = [score for _, score in converted]

    best_score = scores[0]
    in_scope = best_score >= threshold

    logger.info(
        "Query: '%s'  |  best_score=%.4f  threshold=%.2f  in_scope=%s",
        query[:60],
        best_score,
        threshold,
        in_scope,
    )
    for i, (doc, sc) in enumerate(converted):
        logger.debug(
            "  [%d] score=%.4f  source=%s  page=%s  text=%.80s…",
            i,
            sc,
            doc.metadata.get("source", "?"),
            doc.metadata.get("page", "?"),
            doc.page_content.replace("\n", " "),
        )

    return RetrievalResult(in_scope=in_scope, chunks=chunks, scores=scores)
