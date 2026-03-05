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
    def _call_with_retry(self, prompt: str) -> str:
        """Call the Ollama /api/generate endpoint with retry logic."""
        import requests  # lazily imported so non-Gemma users don't need it

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json()["response"].strip()
            except Exception as exc:
                if attempt < self._max_retries:
                    wait = self._retry_base_wait * attempt
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s  – retrying in %ds",
                        attempt, self._max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Ollama ({self.model_name}) failed after {self._max_retries} "
                        f"attempts: {exc}"
                    ) from exc
        # unreachable, but keeps type-checkers happy
        return ""

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
