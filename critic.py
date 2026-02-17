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

logger = logging.getLogger(__name__)


# ── value object ──────────────────────────────────────────────────────
@dataclass
class CriticResult:
    confidence: float          # 0-100
    explanation: str           # brief rationale from the LLM


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
        self._model = genai.GenerativeModel(self.model_name)
        self._max_retries = 3
        self._retry_base_wait = 25          # seconds (Gemini free-tier resets every ~60 s)
        logger.info("GeminiCritic initialised  (model=%s)", self.model_name)

    # ── rate-limit-aware API call ─────────────────────────────────────
    def _call_with_retry(self, prompt: str) -> str:
        """Call Gemini with automatic retry on 429 rate-limit errors."""
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._model.generate_content(prompt)
                return response.text.strip()
            except Exception as exc:
                if "429" in str(exc) or "ResourceExhausted" in type(exc).__name__:
                    wait = self._retry_base_wait * attempt
                    logger.warning(
                        "Rate-limited by Gemini API (attempt %d/%d). "
                        "Retrying in %ds...",
                        attempt, self._max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        # final attempt — let it raise if it fails
        response = self._model.generate_content(prompt)
        return response.text.strip()

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
        """Best-effort JSON parse with regex fallback."""
        # Strip markdown code fences if present
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
            # Fallback: try to pull a number out of the string
            match = re.search(r'"?confidence"?\s*:\s*(\d+)', raw)
            conf = float(match.group(1)) if match else 0.0
            return CriticResult(confidence=conf, explanation=raw[:200])

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
