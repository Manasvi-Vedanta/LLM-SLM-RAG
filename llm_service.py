"""
llm_service.py – Shared LLM call utilities
--------------------------------------------
Provides a single interface for calling Ollama (local) and Gemini (cloud)
models. Used by the Critic for validation and by new pipeline components
(path validator, question generator, etc.) for generation tasks.

Design
~~~~~~
* ``call_ollama``  – low-level Ollama REST call with retry logic.
* ``call_gemini``  – low-level Gemini SDK call with rate-limit retry + fallback chain.
* ``call_llm``     – high-level dispatcher: routes "generation" tasks to the
                     base model and "validation" tasks to the fine-tuned model.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Ollama (local)
# ──────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    max_retries: int = 3,
    retry_base_wait: int = 5,
    timeout: int = 120,
) -> str:
    """Call the Ollama ``/api/generate`` endpoint with retry logic.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    model : str, optional
        Ollama model name. Defaults to ``config.OLLAMA_BASE_MODEL``.
    base_url : str, optional
        Ollama server URL. Defaults to ``config.OLLAMA_BASE_URL``.
    max_retries : int
        Number of retry attempts on failure.
    retry_base_wait : int
        Base wait (seconds) between retries — multiplied by attempt number.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        The model's response text, stripped.
    """
    import requests

    model = model or config.OLLAMA_BASE_MODEL
    url = (base_url or config.OLLAMA_BASE_URL).rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as exc:
            if attempt < max_retries:
                wait = retry_base_wait * attempt
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s – retrying in %ds",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama ({model}) failed after {max_retries} attempts: {exc}"
                ) from exc

    return ""  # unreachable, keeps type-checkers happy


# ──────────────────────────────────────────────
# Gemini (cloud)
# ──────────────────────────────────────────────

def call_gemini(
    prompt: str,
    model_name: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_base_wait: int = 25,
) -> str:
    """Call Google Gemini with rate-limit retry and fallback chain.

    Tries the primary model first, then each model in
    ``config.GEMINI_FALLBACK_MODELS``.  Only 429 / ResourceExhausted
    errors trigger fallback; other exceptions propagate immediately.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    model_name : str, optional
        Primary model. Defaults to ``config.GEMINI_MODEL_NAME``.
    api_key : str, optional
        Gemini API key. Defaults to ``config.GEMINI_API_KEY``.

    Returns
    -------
    str
        The model's response text, stripped.
    """
    import google.generativeai as genai  # type: ignore

    api_key = api_key or config.GEMINI_API_KEY
    if api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("No GEMINI_API_KEY configured.")

    genai.configure(api_key=api_key)
    primary = model_name or config.GEMINI_MODEL_NAME
    fallbacks = getattr(config, "GEMINI_FALLBACK_MODELS", [])
    model_chain = [primary] + list(fallbacks)

    last_exc: Exception | None = None

    for name in model_chain:
        model_obj = genai.GenerativeModel(name)
        for attempt in range(1, max_retries + 1):
            try:
                response = model_obj.generate_content(prompt)
                return response.text.strip()
            except Exception as exc:
                if "429" in str(exc) or "ResourceExhausted" in type(exc).__name__:
                    last_exc = exc
                    wait = retry_base_wait * attempt
                    logger.warning(
                        "Gemini rate-limited by %s (attempt %d/%d). Retrying in %ds...",
                        name, attempt, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

        logger.warning(
            "All %d retries exhausted for %s — trying next model.", max_retries, name
        )

    raise RuntimeError(
        f"All Gemini models rate-limited after exhausting retries: {model_chain}"
    ) from last_exc


# ──────────────────────────────────────────────
# High-level dispatcher
# ──────────────────────────────────────────────

def call_llm(
    prompt: str,
    task: str = "generation",
    fallback_to_gemini: bool = True,
) -> str:
    """Route a prompt to the appropriate LLM.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    task : str
        ``"generation"`` uses the base Gemma model (path correction,
        question generation, content generation, resource suggestions).
        ``"validation"`` uses the fine-tuned critic model.
    fallback_to_gemini : bool
        If True and the Ollama call fails, retry with Gemini cloud.

    Returns
    -------
    str
        The model's response text.
    """
    if task == "validation":
        model = config.OLLAMA_MODEL_NAME
    else:
        model = config.OLLAMA_BASE_MODEL

    try:
        return call_ollama(prompt, model=model)
    except RuntimeError:
        if fallback_to_gemini:
            logger.warning(
                "Ollama (%s) failed for task=%s — falling back to Gemini.", model, task
            )
            return call_gemini(prompt)
        raise


# ──────────────────────────────────────────────
# JSON parsing utility
# ──────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    """Best-effort JSON extraction from LLM output.

    Handles markdown fences, leading/trailing text around the JSON object.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract first JSON object
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to extract JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {raw[:200]}")
