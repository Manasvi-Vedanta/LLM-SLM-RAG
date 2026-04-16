"""
question_generator.py – Document-grounded quiz generation
----------------------------------------------------------
Generates quizzes from a session's FAISS-indexed content.
Each candidate question is validated by the Critic to ensure it's
actually answerable from the source material. Questions that can't be
grounded are supplemented with LLM general-knowledge questions (clearly
labeled).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List

from langchain_community.vectorstores import FAISS

import config
from vector_store import load_session_vectorstore, retrieve
from critic import BaseCritic

logger = logging.getLogger(__name__)


# ── data structures ──────────────────────────────────────────────────

@dataclass
class QuizQuestion:
    question_text: str
    question_type: str                # "mcq" or "open"
    options: dict | None              # {"A": ..., "B": ..., "C": ..., "D": ...} for mcq
    correct_answer: str
    difficulty: str                   # "easy", "medium", "hard"
    explanation: str
    source: str                       # "your_materials" or "general_knowledge"
    source_excerpt: str = ""          # the chunk it was generated from (for grading)
    reference_text: str = ""          # excerpt/code/passage shown alongside question
    bloom_level: str = ""             # Bloom's taxonomy cognitive level


# ── Bloom's taxonomy / Zone of Proximal Development ──────────────────
#
# Bloom's six cognitive levels (remember -> create) ascend in complexity.
# Vygotsky's Zone of Proximal Development (ZPD) says learners grow best
# one step above their current mastery: too easy bores, too hard blocks.
# We map observed mastery into a target Bloom window one level above the
# learner's current comfort zone.

BLOOM_LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

BLOOM_DIFFICULTY_MAP = {
    "remember":   "easy",
    "understand": "easy",
    "apply":      "medium",
    "analyze":    "medium",
    "evaluate":   "hard",
    "create":     "hard",
}


def _get_bloom_levels_for_zpd(mastery_score: float | None,
                              session_difficulty: str = "beginner") -> list[str]:
    """Pick a two-level Bloom window that sits one step above current mastery.

    Below 40% mastery -> remember/understand (build foundation).
    40-70% -> understand/apply (consolidate).
    70-85% -> apply/analyze (extend).
    85%+  -> analyze/evaluate or evaluate/create depending on session depth.
    """
    if mastery_score is None:
        base = {"beginner": ["remember", "understand"],
                "intermediate": ["understand", "apply"],
                "advanced": ["apply", "analyze"]}.get(session_difficulty,
                                                      ["understand", "apply"])
        return base

    if mastery_score < 40:
        return ["remember", "understand"]
    if mastery_score < 70:
        return ["understand", "apply"]
    if mastery_score < 85:
        return ["apply", "analyze"]
    # Advanced ZPD: push into higher-order reasoning
    if session_difficulty == "advanced":
        return ["evaluate", "create"]
    return ["analyze", "evaluate"]


@dataclass
class Quiz:
    session_id: str
    session_number: int
    skill_labels: list[str]
    questions: List[QuizQuestion] = field(default_factory=list)
    total_mcq: int = 0
    total_open: int = 0


# ── helpers ──────────────────────────────────────────────────────────

def _sample_diverse_chunks(vectorstore: FAISS, skill_labels: list[str],
                           n_samples: int = 8) -> list:
    """Sample chunks from the vectorstore using skill labels as queries."""
    all_chunks = []
    seen_content = set()

    for skill in skill_labels:
        query = f"Explain {skill}"
        try:
            results = vectorstore.similarity_search(query, k=3)
            for doc in results:
                # Deduplicate by content hash
                content_key = doc.page_content[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_chunks.append(doc)
        except Exception as exc:
            logger.warning("Chunk sampling failed for '%s': %s", skill, exc)

    # Also sample with generic queries for coverage
    for query in ["key concepts", "important principles", "common examples"]:
        try:
            results = vectorstore.similarity_search(query, k=2)
            for doc in results:
                content_key = doc.page_content[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_chunks.append(doc)
        except Exception:
            pass

    # Shuffle and cap
    random.shuffle(all_chunks)
    return all_chunks[:n_samples]


def _validate_question(question: dict, chunk_text: str,
                       critic: BaseCritic) -> bool:
    """Use the Critic to verify a question is answerable from the chunk."""
    try:
        result = critic.validate(question.get("question_text", ""), chunk_text)
        return result.confidence >= config.CONFIDENCE_THRESHOLD
    except Exception as exc:
        logger.warning("Question validation failed: %s", exc)
        return False


def _generate_general_knowledge_questions(
    skill_labels: list[str],
    critic: BaseCritic,
    n: int = 2,
    difficulty: str = "medium",
    bloom_levels: list[str] | None = None,
) -> list[QuizQuestion]:
    """Generate supplementary questions from LLM knowledge (not document-grounded)."""
    import llm_service

    bloom_hint = ""
    if bloom_levels:
        bloom_hint = f"Target Bloom's levels: {', '.join(bloom_levels)}.\n"

    prompt = f"""\
Generate {n} quiz questions about {', '.join(skill_labels)}.
Difficulty: {difficulty}. Mix of MCQ and open-ended.
{bloom_hint}
Return a JSON array where each object has:
{{"question_text": "...", "type": "mcq" or "open",
  "options": {{"A":"..","B":"..","C":"..","D":".."}} or null,
  "correct_answer": "...", "difficulty": "{difficulty}",
  "bloom_level": "<remember|understand|apply|analyze|evaluate|create>",
  "explanation": "..."}}

ONLY output the JSON array:
"""
    try:
        raw = llm_service.call_llm(prompt, task="generation")
        data = llm_service.parse_json_response(raw)
        items = data if isinstance(data, list) else [data]

        questions = []
        for item in items[:n]:
            questions.append(QuizQuestion(
                question_text=item.get("question_text", ""),
                question_type=item.get("type", "open"),
                options=item.get("options"),
                correct_answer=str(item.get("correct_answer", "")),
                difficulty=item.get("difficulty", difficulty),
                explanation=item.get("explanation", ""),
                source="general_knowledge",
                source_excerpt="",
                bloom_level=str(item.get("bloom_level", "")).lower(),
            ))
        return questions
    except Exception as exc:
        logger.warning("General knowledge question generation failed: %s", exc)
        return []


# ── main generator ───────────────────────────────────────────────────

def generate_session_quiz(
    session_id: str,
    session: dict,
    critic: BaseCritic,
    n_mcq: int | None = None,
    n_open: int | None = None,
    current_mastery: float | None = None,
) -> Quiz:
    """Generate a validated quiz for a session.

    Parameters
    ----------
    session_id : str
        The session's vectorstore ID.
    session : dict
        The session dict from the learning path.
    critic : BaseCritic
        Critic instance for validation and question generation.
    n_mcq : int, optional
        Number of MCQ questions (default from config).
    n_open : int, optional
        Number of open-ended questions (default from config).

    Returns
    -------
    Quiz
        A quiz with validated, labeled questions.
    """
    n_mcq = n_mcq if n_mcq is not None else config.QUIZ_MCQ_COUNT
    n_open = n_open if n_open is not None else config.QUIZ_OPEN_COUNT
    total_needed = n_mcq + n_open

    skill_labels = session.get("skills", [])
    session_number = session.get("session_number", 0)
    difficulty = session.get("difficulty_level", "beginner")

    # Compute ZPD-targeted Bloom levels from current mastery so the quiz
    # sits one rung above the learner's current comfort zone.
    bloom_levels = _get_bloom_levels_for_zpd(current_mastery, difficulty)

    logger.info("Generating quiz for session %s (%d MCQ + %d open, ZPD=%s, mastery=%s)",
                session_id, n_mcq, n_open, bloom_levels, current_mastery)

    # Load vectorstore
    try:
        vectorstore = load_session_vectorstore(session_id)
    except FileNotFoundError:
        logger.warning("No vectorstore for session %s — using general knowledge only",
                       session_id)
        gk_questions = _generate_general_knowledge_questions(
            skill_labels, critic, n=total_needed, difficulty=difficulty,
            bloom_levels=bloom_levels,
        )
        return Quiz(
            session_id=session_id,
            session_number=session_number,
            skill_labels=skill_labels,
            questions=gk_questions,
            total_mcq=sum(1 for q in gk_questions if q.question_type == "mcq"),
            total_open=sum(1 for q in gk_questions if q.question_type == "open"),
        )

    # Sample diverse chunks
    chunks = _sample_diverse_chunks(vectorstore, skill_labels)

    if not chunks:
        logger.warning("No chunks found for session %s", session_id)
        gk_questions = _generate_general_knowledge_questions(
            skill_labels, critic, n=total_needed, difficulty=difficulty,
            bloom_levels=bloom_levels,
        )
        return Quiz(
            session_id=session_id,
            session_number=session_number,
            skill_labels=skill_labels,
            questions=gk_questions,
            total_mcq=sum(1 for q in gk_questions if q.question_type == "mcq"),
            total_open=sum(1 for q in gk_questions if q.question_type == "open"),
        )

    # Generate candidate questions from chunks
    validated_questions: list[QuizQuestion] = []
    candidate_texts: set[str] = set()  # for deduplication

    for chunk in chunks:
        if len(validated_questions) >= total_needed:
            break

        # Ask critic to generate questions from this chunk
        primary_skill = skill_labels[0] if skill_labels else "the topic"
        raw_questions = critic.generate_questions(
            excerpt=chunk.page_content,
            skill_label=primary_skill,
            difficulty=difficulty,
            n=2,
            bloom_levels=bloom_levels,
        )

        for rq in raw_questions:
            if len(validated_questions) >= total_needed:
                break

            q_text = rq.get("question_text", "")
            if not q_text or q_text in candidate_texts:
                continue
            candidate_texts.add(q_text)

            # Validate that the question is answerable from the chunk
            if _validate_question(rq, chunk.page_content, critic):
                validated_questions.append(QuizQuestion(
                    question_text=q_text,
                    question_type=rq.get("type", "open"),
                    options=rq.get("options"),
                    correct_answer=str(rq.get("correct_answer", "")),
                    difficulty=rq.get("difficulty", difficulty),
                    explanation=rq.get("explanation", ""),
                    source="your_materials",
                    source_excerpt=chunk.page_content,
                    reference_text=rq.get("reference_text", ""),
                    bloom_level=str(rq.get("bloom_level", "")).lower(),
                ))

    # Supplement with general knowledge if not enough document-grounded questions
    shortfall = total_needed - len(validated_questions)
    if shortfall > 0:
        logger.info("Supplementing with %d general knowledge questions", shortfall)
        gk_questions = _generate_general_knowledge_questions(
            skill_labels, critic, n=shortfall, difficulty=difficulty
        )
        validated_questions.extend(gk_questions)

    # Balance MCQ vs open-ended
    mcq_qs = [q for q in validated_questions if q.question_type == "mcq"]
    open_qs = [q for q in validated_questions if q.question_type == "open"]

    # Trim to target counts
    final_questions = mcq_qs[:n_mcq] + open_qs[:n_open]

    # If we don't have enough of one type, fill from the other
    if len(final_questions) < total_needed:
        remaining = [q for q in validated_questions if q not in final_questions]
        final_questions.extend(remaining[:total_needed - len(final_questions)])

    random.shuffle(final_questions)

    quiz = Quiz(
        session_id=session_id,
        session_number=session_number,
        skill_labels=skill_labels,
        questions=final_questions,
        total_mcq=sum(1 for q in final_questions if q.question_type == "mcq"),
        total_open=sum(1 for q in final_questions if q.question_type == "open"),
    )

    logger.info(
        "Quiz generated: %d questions (%d MCQ, %d open, %d from materials, %d general)",
        len(quiz.questions), quiz.total_mcq, quiz.total_open,
        sum(1 for q in quiz.questions if q.source == "your_materials"),
        sum(1 for q in quiz.questions if q.source == "general_knowledge"),
    )

    return quiz
