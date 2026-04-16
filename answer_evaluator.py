"""
answer_evaluator.py – Quiz grading and answer evaluation
---------------------------------------------------------
Grades MCQ answers directly and open-ended answers via the Critic's
evaluation prompt against source excerpts. Computes per-skill mastery
scores and saves results to the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import config
import database
from vector_store import load_session_vectorstore, retrieve
from critic import BaseCritic
from question_generator import Quiz, QuizQuestion

logger = logging.getLogger(__name__)


# ── data structures ──────────────────────────────────────────────────

@dataclass
class AnswerResult:
    question_text: str
    question_type: str
    question_source: str          # "your_materials" or "general_knowledge"
    user_answer: str
    correct_answer: str
    score: float                  # 0-100
    max_score: float              # always 100
    feedback: str


@dataclass
class QuizResult:
    session_id: str
    session_number: int
    attempt_id: int | None        # DB ID after saving
    answers: List[AnswerResult] = field(default_factory=list)
    total_score: float = 0.0
    max_score: float = 0.0
    percentage: float = 0.0
    per_skill_scores: dict = field(default_factory=dict)
    # per_skill_scores: {"Python": 85.0, "Data Structures": 60.0}


# ── grading functions ────────────────────────────────────────────────

def evaluate_mcq(question: QuizQuestion, user_choice: str) -> AnswerResult:
    """Grade an MCQ answer by direct comparison."""
    correct = question.correct_answer.strip().upper()
    chosen = user_choice.strip().upper()
    is_correct = chosen == correct

    score = 100.0 if is_correct else 0.0
    if is_correct:
        feedback = f"Correct! {question.explanation}"
    else:
        feedback = (
            f"Incorrect. The correct answer is {correct}. "
            f"{question.explanation}"
        )

    return AnswerResult(
        question_text=question.question_text,
        question_type="mcq",
        question_source=question.source,
        user_answer=user_choice,
        correct_answer=question.correct_answer,
        score=score,
        max_score=100.0,
        feedback=feedback,
    )


def evaluate_open_answer(
    question: QuizQuestion,
    user_answer: str,
    session_id: str,
    critic: BaseCritic,
) -> AnswerResult:
    """Grade an open-ended answer using the Critic.

    If the question is from materials, retrieves the relevant chunk
    and uses it as the reference. For general knowledge questions,
    uses the stored correct_answer as reference.
    """
    reference = question.source_excerpt or question.correct_answer

    # If we have a session vectorstore, try to find the best matching chunk
    if question.source == "your_materials" and not question.source_excerpt:
        try:
            vs = load_session_vectorstore(session_id)
            result = retrieve(vs, question.question_text, k=1)
            if result.best_chunk:
                reference = result.best_chunk.page_content
        except Exception:
            pass

    # Use critic to evaluate
    eval_result = critic.evaluate_answer(
        question=question.question_text,
        user_answer=user_answer,
        reference_excerpt=reference,
    )

    score = float(eval_result.get("score", 0))
    feedback = eval_result.get("feedback", "No feedback available.")

    return AnswerResult(
        question_text=question.question_text,
        question_type="open",
        question_source=question.source,
        user_answer=user_answer,
        correct_answer=question.correct_answer,
        score=score,
        max_score=100.0,
        feedback=feedback,
    )


# ── full quiz grading ────────────────────────────────────────────────

def grade_quiz(
    quiz: Quiz,
    user_answers: list[str],
    critic: BaseCritic,
    user_id: int,
    path_id: int,
) -> QuizResult:
    """Grade all answers in a quiz and save results to the database.

    Parameters
    ----------
    quiz : Quiz
        The quiz that was presented to the user.
    user_answers : list[str]
        User's answers in the same order as quiz.questions.
        For MCQ: the letter choice ("A", "B", etc.)
        For open-ended: the free-text answer.
    critic : BaseCritic
        Critic instance for evaluating open-ended answers.
    user_id : int
        Database user ID.
    path_id : int
        Database learning path ID.

    Returns
    -------
    QuizResult
        Full grading results with per-answer feedback.
    """
    if len(user_answers) != len(quiz.questions):
        raise ValueError(
            f"Expected {len(quiz.questions)} answers, got {len(user_answers)}"
        )

    # Determine attempt number
    existing = database.get_quiz_attempts(path_id, quiz.session_number, user_id)
    attempt_number = len(existing) + 1

    # Create attempt record
    attempt_id = database.create_quiz_attempt(
        path_id=path_id,
        session_number=quiz.session_number,
        user_id=user_id,
        attempt_number=attempt_number,
    )

    # Grade each question
    answer_results: list[AnswerResult] = []

    for question, user_answer in zip(quiz.questions, user_answers):
        if question.question_type == "mcq":
            result = evaluate_mcq(question, user_answer)
        else:
            result = evaluate_open_answer(
                question, user_answer, quiz.session_id, critic
            )
        answer_results.append(result)

        # Save to database
        database.save_quiz_answer(
            attempt_id=attempt_id,
            question_text=result.question_text,
            question_type=result.question_type,
            question_source=result.question_source,
            user_answer=result.user_answer,
            correct_answer=result.correct_answer,
            score=result.score,
            max_score=result.max_score,
            feedback=result.feedback,
        )

    # Compute totals
    total_score = sum(r.score for r in answer_results)
    max_score = sum(r.max_score for r in answer_results)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0.0

    # Update attempt with scores
    database.update_quiz_attempt(attempt_id, total_score, max_score, percentage)

    # Compute per-skill scores (average across all questions)
    # All questions in a session quiz map to that session's skills
    per_skill = {}
    for skill in quiz.skill_labels:
        per_skill[skill] = percentage  # same score for all skills in session

    # Update mastery for each skill
    for skill, score in per_skill.items():
        database.upsert_skill_mastery(
            user_id=user_id,
            path_id=path_id,
            skill_label=skill,
            session_number=quiz.session_number,
            new_score=score,
        )

    quiz_result = QuizResult(
        session_id=quiz.session_id,
        session_number=quiz.session_number,
        attempt_id=attempt_id,
        answers=answer_results,
        total_score=total_score,
        max_score=max_score,
        percentage=percentage,
        per_skill_scores=per_skill,
    )

    logger.info(
        "Quiz graded: session %s, attempt %d, score %.1f%% (%d/%d)",
        quiz.session_id, attempt_number, percentage,
        sum(1 for r in answer_results if r.score >= 50),
        len(answer_results),
    )

    return quiz_result
