"""Unified LLM client supporting Ollama (local) and OpenAI."""
from __future__ import annotations

import os
from typing import Optional

import requests


def is_ollama_available(model: str = "mistral:7b") -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        return resp.status_code == 200
    except Exception:
        return False


def _call_ollama(prompt: str, model: str = "mistral:7b", timeout: int = 90) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    import openai  # type: ignore

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


def call_llm(
    prompt: str,
    provider: str = "ollama",
    model: Optional[str] = None,
    timeout: int = 90,
) -> str:
    """Call LLM. Returns empty string on any failure."""
    try:
        if provider == "openai":
            return _call_openai(prompt, model or "gpt-4o-mini")
        return _call_ollama(prompt, model or "mistral:7b", timeout=timeout)
    except Exception:
        return ""


def make_llm_fn(provider: str = "ollama", model: Optional[str] = None):
    """Return a callable(prompt) -> str for passing to other modules."""
    def fn(prompt: str) -> str:
        return call_llm(prompt, provider=provider, model=model)
    return fn
