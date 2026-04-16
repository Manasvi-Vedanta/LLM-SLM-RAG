"""
critic.py  –  The **Critic** (Validator)
-----------------------------------------
An LLM-backed judge that decides whether a retrieved excerpt actually
answers the user's question.

Public API
~~~~~~~~~~
* ``CriticResult`` – dataclass with ``confidence`` (0-100) and ``explanation``.
* ``BaseCritic``   – abstract interface so we can swap LLM backends easily.
* ``GeminiCritic`` – concrete implementation using the Google Gemini API.
* ``MockCritic``   – deterministic stand-in for unit tests / offline work.

Design notes
~~~~~~~~~~~~
* The Critic never rewrites the excerpt — it only *scores* it.
* If confidence is below the threshold the **pipeline** (not the Critic)
  will ask a separate ``generate_fallback_answer`` call to produce a
  response from the LLM's own knowledge.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import config
import llm_service

logger = logging.getLogger(__name__)


# ── value object ──────────────────────────────────────────────────────
@dataclass
class CriticResult:
    confidence: float          # 0-100
    explanation: str           # brief rationale from the LLM


# ── shared JSON parsing ──────────────────────────────────────────────
def parse_critic_json(raw: str) -> CriticResult:
    """Best-effort JSON parse of Critic LLM output with regex fallback."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        return CriticResult(
            confidence=float(data["confidence"]),
            explanation=str(data.get("explanation", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        match = re.search(r'"?confidence"?\s*:\s*(\d+)', raw)
        conf = float(match.group(1)) if match else 0.0
        return CriticResult(confidence=conf, explanation=raw[:200])


# ── post-processing for formatted answers ────────────────────────────
def _clean_formatted_answer(raw: str) -> str:
    """Strip leaked prompt instructions from the formatter output."""
    # Cut at the first "### INSTRUCTIONS" block that the model echoes back
    marker = "### INSTRUCTIONS"
    idx = raw.find(marker)
    if idx != -1:
        raw = raw[:idx]
    # Also cut at "Rules:" if it appears after the main answer body
    # (only if it looks like leaked prompt, i.e. followed by "* " bullets)
    rules_match = re.search(r"\nRules:\n\*\s", raw)
    if rules_match:
        raw = raw[:rules_match.start()]
    return raw.strip()


# ── abstract base ─────────────────────────────────────────────────────
class BaseCritic(ABC):
    """Interface every Critic implementation must follow."""

    @abstractmethod
    def validate(self, question: str, excerpt: str) -> CriticResult:
        """Score how well *excerpt* answers *question*."""
        ...

    @abstractmethod
    def generate_fallback_answer(self, question: str) -> str:
        """Produce a general-knowledge answer when the document fails."""
        ...

    @abstractmethod
    def format_answer(self, question: str, excerpt: str) -> str:
        """Restructure a validated excerpt into a clean, readable answer."""
        ...

    @abstractmethod
    def generate_questions(self, excerpt: str, skill_label: str,
                           difficulty: str, n: int,
                           bloom_levels: list[str] | None = None) -> list[dict]:
        """Generate quiz questions from an excerpt.

        ``bloom_levels`` — optional ZPD-targeted cognitive levels
        (e.g. ["understand", "apply"]) that the generator should bias toward.
        """
        ...


def _build_bloom_hint(bloom_levels: list[str] | None) -> str:
    """Format an optional Bloom's-taxonomy targeting hint for the prompt."""
    if not bloom_levels:
        return ""
    levels = ", ".join(bloom_levels)
    return (
        "### TARGET COGNITIVE LEVELS (Bloom's Taxonomy)\n"
        f"Bias the questions toward these levels: {levels}.\n"
        "Pick verbs that match (remember: list/define; understand: explain/summarize; "
        "apply: use/execute; analyze: compare/differentiate; "
        "evaluate: critique/justify; create: design/compose).\n\n"
    )

    @abstractmethod
    def evaluate_answer(self, question: str, user_answer: str,
                        reference_excerpt: str) -> dict:
        """Grade a user's free-text answer against a reference excerpt."""
        ...


# ── Gemini implementation ─────────────────────────────────────────────
class GeminiCritic(BaseCritic):
    """
    Uses Google's ``generativeai`` SDK to call a Gemini model.

    The constructor lazily configures the SDK the first time it is
    instantiated so that importing this module never raises if the key
    is missing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL_NAME

        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=self.api_key)
        self._genai = genai                  # keep ref for creating fallback models
        self._model = genai.GenerativeModel(self.model_name)
        self._max_retries = 3
        self._retry_base_wait = 25          # seconds (Gemini free-tier resets every ~60 s)

        # Build ordered fallback chain: primary → fallback models
        fallback_names = getattr(config, "GEMINI_FALLBACK_MODELS", [])
        self._model_chain: list[tuple[str, object]] = [
            (self.model_name, self._model),
        ]
        for fb_name in fallback_names:
            self._model_chain.append(
                (fb_name, genai.GenerativeModel(fb_name))
            )

        logger.info(
            "GeminiCritic initialised  (model=%s, fallbacks=%s)",
            self.model_name,
            [n for n, _ in self._model_chain[1:]] or "none",
        )

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """Return True if *exc* looks like a 429 / ResourceExhausted error."""
        return "429" in str(exc) or "ResourceExhausted" in type(exc).__name__

    # ── rate-limit-aware API call with model fallback chain ───────────
    def _call_with_retry(self, prompt: str) -> str:
        """Call Gemini with automatic retry on 429 rate-limit errors.

        Tries each model in ``self._model_chain`` (primary → fallbacks).
        For every model the method retries ``_max_retries`` times before
        moving on to the next model.  Only rate-limit errors trigger a
        fallback; any other exception is raised immediately.
        """
        last_exc: Exception | None = None

        for model_name, model_obj in self._model_chain:
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = model_obj.generate_content(prompt)
                    return response.text.strip()
                except Exception as exc:
                    if self._is_rate_limit(exc):
                        last_exc = exc
                        wait = self._retry_base_wait * attempt
                        logger.warning(
                            "Rate-limited by %s (attempt %d/%d). "
                            "Retrying in %ds...",
                            model_name, attempt, self._max_retries, wait,
                        )
                        time.sleep(wait)
                    else:
                        raise

            # All retries exhausted for this model — cascade to next
            logger.warning(
                "All %d retries exhausted for %s — falling back to next model.",
                self._max_retries, model_name,
            )

        # Every model in the chain is rate-limited
        raise RuntimeError(
            "All Gemini models rate-limited after exhausting retries: "
            f"{[n for n, _ in self._model_chain]}"
        ) from last_exc

    # ── validation prompt ─────────────────────────────────────────────
    _VALIDATE_PROMPT = """\
You are a strict validation judge for a Retrieval-Augmented Generation system.

### TASK
Decide whether the **Excerpt** below correctly and sufficiently answers the
**Question**.  Return your assessment as JSON with exactly two keys:

  {{"confidence": <int 0-100>, "explanation": "<one sentence>"}}

Rules:
* 100 = the excerpt fully and correctly answers the question.
*   0 = the excerpt is completely irrelevant.
* Be harsh — partial or vague matches should score below 70.

### QUESTION
{question}

### EXCERPT
{excerpt}

### YOUR JSON RESPONSE (no markdown fences):
"""

    def validate(self, question: str, excerpt: str) -> CriticResult:
        prompt = self._VALIDATE_PROMPT.format(question=question, excerpt=excerpt)
        raw = self._call_with_retry(prompt)
        logger.debug("Critic raw response: %s", raw)

        return self._parse_validation(raw)

    @staticmethod
    def _parse_validation(raw: str) -> CriticResult:
        """Delegates to module-level ``parse_critic_json``."""
        return parse_critic_json(raw)

    # ── fallback generation ───────────────────────────────────────────
    _FALLBACK_PROMPT = """\
The user asked the following question, but the reference documents did not
contain a sufficient answer.

Question: {question}

Please provide a helpful, accurate answer using your own general knowledge.
Begin your answer with:
"The document did not contain the answer, but here is the correct answer
based on general knowledge..."
"""

    def generate_fallback_answer(self, question: str) -> str:
        prompt = self._FALLBACK_PROMPT.format(question=question)
        return self._call_with_retry(prompt)

    # ── answer formatting ────────────────────────────────────────────
    _FORMAT_PROMPT = """\
You are a teaching assistant that restructures raw textbook excerpts into
clear, student-friendly answers.  You may ONLY use information present in
the Excerpt — do NOT add outside knowledge.

### QUESTION
{question}

### EXCERPT
{excerpt}

### INSTRUCTIONS
Rewrite the excerpt into a structured answer using **exactly** this format
(omit any section that does not apply):

**Definition:**
A clear 2-3 sentence explanation answering the question.

**Syntax / Formula:**
Code block or mathematical notation (only if relevant).

**Key Points:**
- Bullet 1
- Bullet 2
- Bullet 3

**Example:**
A concrete example from the excerpt (only if one exists).

Rules:
* Keep it concise — no filler, no repetition.
* Use simple language a beginner can understand.
* If the excerpt contains code, preserve it exactly in a code block.
* Do NOT invent information not in the excerpt.
"""

    def format_answer(self, question: str, excerpt: str) -> str:
        prompt = self._FORMAT_PROMPT.format(question=question, excerpt=excerpt)
        raw = self._call_with_retry(prompt)
        return _clean_formatted_answer(raw)

    # ── question generation ──────────────────────────────────────────
    _GENERATE_QUESTIONS_PROMPT = """\
You are an expert quiz creator for an educational platform.

### TASK
Generate exactly {n} quiz questions based on the excerpt below.
The questions should test understanding of "{skill_label}" at {difficulty} level.

Create a mix of question types:
- MCQ (multiple choice with 4 options, one correct)
- OPEN (open-ended, short answer)

{bloom_hint}### EXCERPT
{excerpt}

### OUTPUT FORMAT
Return a JSON array of question objects. Each object must have:
{{
  "question_text": "<the question>",
  "type": "mcq" or "open",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}} (only for mcq, null for open),
  "correct_answer": "<letter for mcq, or short answer text for open>",
  "difficulty": "{difficulty}",
  "bloom_level": "<one of: remember, understand, apply, analyze, evaluate, create>",
  "explanation": "<why this is the correct answer>",
  "reference_text": "<the specific part of the excerpt the question is about — include any code, formula, or passage the student needs to see to answer>"
}}

Rules:
* Questions MUST be answerable from the excerpt
* MCQ distractors should be plausible but clearly wrong
* Open-ended answers should be concise (1-3 sentences)
* NEVER use phrases like "from the excerpt", "in the passage above", "from the code below", "according to the text". Instead, if a question depends on specific code, a formula, or a text passage, include that material in the "reference_text" field and write the question as if the reference_text will be shown alongside it
* ONLY output the JSON array. No extra text.

### YOUR JSON RESPONSE:
"""

    def generate_questions(self, excerpt: str, skill_label: str,
                           difficulty: str = "medium", n: int = 3,
                           bloom_levels: list[str] | None = None) -> list[dict]:
        prompt = self._GENERATE_QUESTIONS_PROMPT.format(
            excerpt=excerpt, skill_label=skill_label,
            difficulty=difficulty, n=n,
            bloom_hint=_build_bloom_hint(bloom_levels),
        )
        raw = self._call_with_retry(prompt)
        try:
            result = llm_service.parse_json_response(raw)
            if isinstance(result, list):
                return result
            return result.get("questions", [result])
        except (ValueError, AttributeError):
            logger.warning("Failed to parse generated questions: %s", raw[:200])
            return []

    # ── answer evaluation ────────────────────────────────────────────
    _EVALUATE_ANSWER_PROMPT = """\
You are an expert educational assessor.

### TASK
Evaluate a student's answer to a quiz question by comparing it against
the reference material.

### QUESTION
{question}

### STUDENT'S ANSWER
{user_answer}

### REFERENCE EXCERPT
{reference_excerpt}

### OUTPUT FORMAT
Return a JSON object with exactly these keys:
{{"score": <int 0-100>, "feedback": "<2-3 sentences of constructive feedback>"}}

Scoring guide:
* 90-100: Excellent — demonstrates thorough understanding
* 70-89: Good — mostly correct with minor gaps
* 50-69: Partial — shows some understanding but missing key points
* 25-49: Weak — significant misunderstanding or incomplete
* 0-24: Incorrect — answer is wrong or irrelevant

Rules:
* Be fair but rigorous
* Focus feedback on what was right, what was wrong, and what to review
* ONLY output the JSON object. No extra text.

### YOUR JSON RESPONSE:
"""

    def evaluate_answer(self, question: str, user_answer: str,
                        reference_excerpt: str) -> dict:
        prompt = self._EVALUATE_ANSWER_PROMPT.format(
            question=question, user_answer=user_answer,
            reference_excerpt=reference_excerpt,
        )
        raw = self._call_with_retry(prompt)
        try:
            return llm_service.parse_json_response(raw)
        except ValueError:
            logger.warning("Failed to parse answer evaluation: %s", raw[:200])
            return {"score": 0, "feedback": "Evaluation failed — could not parse LLM output."}


# ── Gemma implementation (local SLM via Ollama) ──────────────────────
class GemmaCritic(BaseCritic):
    """
    Uses a locally-running Gemma 3 4B model via Ollama.

    Prerequisites
    ~~~~~~~~~~~~~
    1. Install Ollama:  https://ollama.com
    2. Pull the model:  ``ollama pull gemma3:4b``
    3. Set ``CRITIC_BACKEND = "gemma"`` in ``config.py``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name or config.OLLAMA_MODEL_NAME
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._max_retries = 3
        self._retry_base_wait = 5  # seconds
        logger.info(
            "GemmaCritic initialised  (model=%s, url=%s)",
            self.model_name,
            self.base_url,
        )

    # ── Ollama REST call ──────────────────────────────────────────────
    def _call_ollama(self, prompt: str, model: str | None = None) -> str:
        """Call the Ollama /api/generate endpoint via shared llm_service."""
        return llm_service.call_ollama(
            prompt,
            model=model or self.model_name,
            base_url=self.base_url,
            max_retries=self._max_retries,
            retry_base_wait=self._retry_base_wait,
        )

    def _call_with_retry(self, prompt: str) -> str:
        """Call using the fine-tuned critic model."""
        return self._call_ollama(prompt)

    # ── validation prompt ─────────────────────────────────────────────
    _VALIDATE_PROMPT = """\
You are a strict validation judge for a Retrieval-Augmented Generation system.

### TASK
Decide whether the **Excerpt** below correctly and sufficiently answers the
**Question**.  Return your assessment as JSON with exactly two keys:

  {{"confidence": <int 0-100>, "explanation": "<one sentence>"}}

Rules:
* 100 = the excerpt fully and correctly answers the question.
*   0 = the excerpt is completely irrelevant.
* Be harsh — partial or vague matches should score below 70.
* ONLY output the JSON object. No extra text.

### QUESTION
{question}

### EXCERPT
{excerpt}

### YOUR JSON RESPONSE:
"""

    def validate(self, question: str, excerpt: str) -> CriticResult:
        prompt = self._VALIDATE_PROMPT.format(question=question, excerpt=excerpt)
        raw = self._call_with_retry(prompt)
        logger.debug("GemmaCritic raw response: %s", raw)
        return parse_critic_json(raw)

    # ── fallback generation ───────────────────────────────────────────
    _FALLBACK_PROMPT = """\
The user asked the following question, but the reference documents did not
contain a sufficient answer.

Question: {question}

Please provide a helpful, accurate answer using your own general knowledge.
Begin your answer with:
"The document did not contain the answer, but here is the correct answer
based on general knowledge..."
"""

    def generate_fallback_answer(self, question: str) -> str:
        prompt = self._FALLBACK_PROMPT.format(question=question)
        return self._call_with_retry(prompt)

    # ── answer formatting ────────────────────────────────────────────
    _FORMAT_PROMPT = """\
You are a teaching assistant. Restructure the excerpt below into a clean,
student-friendly answer. ONLY use information in the excerpt — no outside
knowledge. Output ONLY the formatted answer, nothing else.

Question: {question}

Excerpt: {excerpt}

Format (omit sections that don't apply):

**Definition:**
2-3 sentence explanation.

**Syntax / Formula:**
Code or math notation (if relevant).

**Key Points:**
- Point 1
- Point 2
- Point 3

**Example:**
Concrete example (if present in excerpt).

ANSWER:
"""

    def format_answer(self, question: str, excerpt: str) -> str:
        prompt = self._FORMAT_PROMPT.format(question=question, excerpt=excerpt)
        # Use the base Gemma model for formatting — the fine-tuned critic
        # model is trained to output JSON validation, not prose.
        raw = self._call_ollama(prompt, model=config.OLLAMA_BASE_MODEL)
        return _clean_formatted_answer(raw)

    # ── question generation (uses base model) ────────────────────────
    def generate_questions(self, excerpt: str, skill_label: str,
                           difficulty: str = "medium", n: int = 3,
                           bloom_levels: list[str] | None = None) -> list[dict]:
        prompt = GeminiCritic._GENERATE_QUESTIONS_PROMPT.format(
            excerpt=excerpt, skill_label=skill_label,
            difficulty=difficulty, n=n,
            bloom_hint=_build_bloom_hint(bloom_levels),
        )
        raw = self._call_ollama(prompt, model=config.OLLAMA_BASE_MODEL)
        try:
            result = llm_service.parse_json_response(raw)
            if isinstance(result, list):
                return result
            return result.get("questions", [result])
        except (ValueError, AttributeError):
            logger.warning("Failed to parse generated questions: %s", raw[:200])
            return []

    # ── answer evaluation (uses base model) ──────────────────────────
    def evaluate_answer(self, question: str, user_answer: str,
                        reference_excerpt: str) -> dict:
        prompt = GeminiCritic._EVALUATE_ANSWER_PROMPT.format(
            question=question, user_answer=user_answer,
            reference_excerpt=reference_excerpt,
        )
        raw = self._call_ollama(prompt, model=config.OLLAMA_BASE_MODEL)
        try:
            return llm_service.parse_json_response(raw)
        except ValueError:
            logger.warning("Failed to parse answer evaluation: %s", raw[:200])
            return {"score": 0, "feedback": "Evaluation failed — could not parse LLM output."}


# ── Mock implementation (offline / tests) ────────────────────────────
class MockCritic(BaseCritic):
    """
    Deterministic critic that always returns a fixed confidence.
    Useful for unit tests and running without an API key.
    """

    def __init__(self, fixed_confidence: float = 90.0):
        self.fixed_confidence = fixed_confidence
        logger.info("MockCritic initialised (confidence=%.1f)", fixed_confidence)

    def validate(self, question: str, excerpt: str) -> CriticResult:
        return CriticResult(
            confidence=self.fixed_confidence,
            explanation="[MockCritic] Fixed confidence score.",
        )

    def generate_fallback_answer(self, question: str) -> str:
        return (
            "The document did not contain the answer, but here is the "
            "correct answer based on general knowledge...\n\n"
            "[MockCritic] This is a placeholder fallback answer for: "
            f"{question}"
        )

    def format_answer(self, question: str, excerpt: str) -> str:
        return f"**Definition:**\n{excerpt[:300]}"

    def generate_questions(self, excerpt: str, skill_label: str,
                           difficulty: str = "medium", n: int = 3,
                           bloom_levels: list[str] | None = None) -> list[dict]:
        levels = bloom_levels or ["understand", "apply"]
        questions = []
        for i in range(n):
            q_type = "mcq" if i < n // 2 + 1 else "open"
            q = {
                "question_text": f"[MockCritic] Sample {q_type} question {i+1} about {skill_label}",
                "type": q_type,
                "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"} if q_type == "mcq" else None,
                "correct_answer": "A" if q_type == "mcq" else f"Sample answer about {skill_label}",
                "difficulty": difficulty,
                "bloom_level": levels[i % len(levels)],
                "explanation": "[MockCritic] This is a placeholder question.",
            }
            questions.append(q)
        return questions

    def evaluate_answer(self, question: str, user_answer: str,
                        reference_excerpt: str) -> dict:
        return {"score": 75, "feedback": "[MockCritic] Placeholder evaluation."}


# ── factory ──────────────────────────────────────────────────────────
def create_critic(backend: str | None = None) -> BaseCritic:
    """
    Instantiate the right Critic based on ``config.CRITIC_BACKEND``.

    Supported values: ``"gemini"``, ``"gemma"``, ``"mock"``.
    """
    backend = (backend or config.CRITIC_BACKEND).lower().strip()

    if backend == "gemini":
        if config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            logger.warning(
                "No GEMINI_API_KEY set — falling back to MockCritic."
            )
            return MockCritic(fixed_confidence=90.0)
        return GeminiCritic()

    if backend == "gemma":
        return GemmaCritic()

    if backend == "mock":
        return MockCritic(fixed_confidence=90.0)

    raise ValueError(
        f"Unknown CRITIC_BACKEND: {backend!r}.  "
        f"Supported: 'gemini', 'gemma', 'mock'."
    )
