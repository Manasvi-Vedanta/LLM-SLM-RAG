"""
pipeline.py  –  RAG Pipeline Orchestrator
-------------------------------------------
Ties together the **Actor** (vector_store) and the **Critic** (critic)
into the full Self-Correcting RAG loop described in the architecture.

Flow
~~~~
1. ``retrieve`` top-k chunks (Actor).
2. **Scope Gate** – if best similarity < threshold → "Out of Scope".
3. ``validate`` the excerpt via the Critic LLM.
4. **Confidence Gate**:
   - ≥ 85 %  → return the *exact* retrieved excerpt (no LLM rewriting).
   - < 85 %  → Critic generates a fallback from its own knowledge.

Public API
~~~~~~~~~~
* ``RAGPipeline``  – main class; instantiate once, call ``query()`` many times.
* ``QueryResult``  – dataclass capturing everything about one query cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_community.vectorstores import FAISS

import config
from vector_store import RetrievalResult, retrieve, load_session_vectorstore
from critic import BaseCritic, CriticResult

logger = logging.getLogger(__name__)


# ── value object ──────────────────────────────────────────────────────
@dataclass
class QueryResult:
    """Everything we want to surface about a single query cycle."""

    question: str
    answer: str
    source: str                          # "document" | "general_knowledge" | "out_of_scope"
    retrieval: Optional[RetrievalResult] = None
    critic_result: Optional[CriticResult] = None
    metadata: dict = field(default_factory=dict)


# ── orchestrator ──────────────────────────────────────────────────────
class RAGPipeline:
    """
    Self-Correcting RAG pipeline (Actor-Critic).

    Parameters
    ----------
    vectorstore : FAISS
        The already-built / loaded FAISS index (Actor).
    critic : BaseCritic
        Any Critic implementation (Gemini, Mock, future models …).
    top_k : int, optional
        Number of chunks to retrieve (default from ``config``).
    similarity_threshold : float, optional
        Scope gate cut-off (default from ``config``).
    confidence_threshold : float, optional
        Critic confidence cut-off (default from ``config``).
    """

    def __init__(
        self,
        vectorstore: FAISS,
        critic: BaseCritic,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        confidence_threshold: float | None = None,
    ):
        self.vectorstore = vectorstore
        self.critic = critic
        self.top_k = top_k or config.TOP_K
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else config.SIMILARITY_THRESHOLD
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.CONFIDENCE_THRESHOLD
        )

    # ── main entry point ──────────────────────────────────────────────
    def query(self, question: str) -> QueryResult:
        """Run the full Actor → Scope Gate → Critic → Confidence Gate loop."""

        # ---- Step 1: Actor retrieves chunks --------------------------------
        retrieval = retrieve(
            self.vectorstore,
            question,
            k=self.top_k,
            threshold=self.similarity_threshold,
        )

        # ---- Step 2: Scope gate --------------------------------------------
        if not retrieval.in_scope:
            logger.info("⛔ Out of scope (best_score=%s)", retrieval.best_score)
            return QueryResult(
                question=question,
                answer="Out of Scope. The provided documents do not contain "
                       "information relevant to your question.",
                source="out_of_scope",
                retrieval=retrieval,
                metadata={
                    "best_score": retrieval.best_score,
                    "threshold": self.similarity_threshold,
                },
            )

        # ---- Step 3: Critic validates the excerpt --------------------------
        # Combine all retrieved chunks that passed the threshold into one
        # excerpt so the Critic has richer context to validate against.
        passing_excerpts = []
        for chunk, score in zip(retrieval.chunks, retrieval.scores):
            if score >= self.similarity_threshold:
                passing_excerpts.append(chunk.page_content)
        excerpt = "\n\n---\n\n".join(passing_excerpts) if passing_excerpts else retrieval.best_chunk.page_content  # type: ignore[union-attr]

        critic_result = self.critic.validate(question, excerpt)
        logger.info(
            "Critic confidence=%.1f%%  explanation=%s",
            critic_result.confidence,
            critic_result.explanation[:120],
        )

        # ---- Step 4: Confidence gate ---------------------------------------
        if critic_result.confidence > self.confidence_threshold:
            # HIGH confidence → format the excerpt into a structured answer
            source_file = retrieval.best_chunk.metadata.get("source", "unknown")  # type: ignore[union-attr]
            page = retrieval.best_chunk.metadata.get("page", "?")  # type: ignore[union-attr]
            logger.info("✅ High confidence – formatting document excerpt.")
            formatted_answer = self.critic.format_answer(question, excerpt)
            return QueryResult(
                question=question,
                answer=formatted_answer,
                source="document",
                retrieval=retrieval,
                critic_result=critic_result,
                metadata={
                    "source_file": source_file,
                    "page": page,
                    "similarity_score": retrieval.best_score,
                    "confidence": critic_result.confidence,
                },
            )
        else:
            # LOW confidence → Critic generates from its own knowledge
            logger.info("⚠️ Low confidence – falling back to LLM general knowledge.")
            fallback_answer = self.critic.generate_fallback_answer(question)
            return QueryResult(
                question=question,
                answer=fallback_answer,
                source="general_knowledge",
                retrieval=retrieval,
                critic_result=critic_result,
                metadata={
                    "similarity_score": retrieval.best_score,
                    "confidence": critic_result.confidence,
                },
            )

    # ── session-scoped query ─────────────────────────────────────────
    def session_query(self, session_id: str, question: str) -> QueryResult:
        """Run the RAG pipeline against a session-specific vector store.

        Loads the session's FAISS index, runs the same Actor → Scope Gate →
        Critic → Confidence Gate flow, and adds session metadata to the result.
        """
        logger.info("Session query [%s]: %s", session_id, question[:80])

        session_vs = load_session_vectorstore(session_id)

        # Reuse the same pipeline logic but with the session vectorstore
        retrieval = retrieve(
            session_vs,
            question,
            k=self.top_k,
            threshold=self.similarity_threshold,
        )

        # Scope gate
        if not retrieval.in_scope:
            logger.info("Session %s: out of scope (best_score=%s)",
                        session_id, retrieval.best_score)
            return QueryResult(
                question=question,
                answer="This question doesn't seem to be covered in the "
                       "current session's materials.",
                source="out_of_scope",
                retrieval=retrieval,
                metadata={
                    "session_id": session_id,
                    "best_score": retrieval.best_score,
                    "threshold": self.similarity_threshold,
                },
            )

        # Build excerpt from passing chunks
        passing_excerpts = []
        for chunk, score in zip(retrieval.chunks, retrieval.scores):
            if score >= self.similarity_threshold:
                passing_excerpts.append(chunk.page_content)
        excerpt = (
            "\n\n---\n\n".join(passing_excerpts)
            if passing_excerpts
            else retrieval.best_chunk.page_content  # type: ignore[union-attr]
        )

        # Critic validation
        critic_result = self.critic.validate(question, excerpt)
        logger.info("Session %s: critic confidence=%.1f%%",
                    session_id, critic_result.confidence)

        # Build source attribution from best chunk metadata
        best_meta = retrieval.best_chunk.metadata if retrieval.best_chunk else {}  # type: ignore[union-attr]
        source_info = best_meta.get("source", best_meta.get("resource_name", "unknown"))
        source_type = best_meta.get("source_type", "unknown")

        # Confidence gate
        if critic_result.confidence > self.confidence_threshold:
            formatted_answer = self.critic.format_answer(question, excerpt)
            return QueryResult(
                question=question,
                answer=formatted_answer,
                source="document",
                retrieval=retrieval,
                critic_result=critic_result,
                metadata={
                    "session_id": session_id,
                    "source_file": source_info,
                    "source_type": source_type,
                    "page": best_meta.get("page", ""),
                    "similarity_score": retrieval.best_score,
                    "confidence": critic_result.confidence,
                },
            )
        else:
            fallback_answer = self.critic.generate_fallback_answer(question)
            return QueryResult(
                question=question,
                answer=fallback_answer,
                source="general_knowledge",
                retrieval=retrieval,
                critic_result=critic_result,
                metadata={
                    "session_id": session_id,
                    "similarity_score": retrieval.best_score,
                    "confidence": critic_result.confidence,
                },
            )
