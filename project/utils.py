from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:7b"

COMMON_SKILLS = [
    "python",
    "sql",
    "excel",
    "tableau",
    "power bi",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "nlp",
    "llm",
    "rag",
    "fastapi",
    "flask",
    "docker",
    "git",
    "aws",
    "gcp",
    "azure",
    "kubernetes",
    "airflow",
    "dbt",
    "spark",
    "postgresql",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_filename(name: str, max_len: int = 100) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return (clean or "report")[:max_len]


def guess_company(text: str, url: str = "") -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines[:12]:
        if len(ln.split()) <= 8 and re.search(r"(gmbh|ag|inc|llc|ltd|company|startup)", ln, re.I):
            return ln
    if url:
        host = re.sub(r"^https?://", "", url).split("/")[0]
        host = host.replace("www.", "")
        base = host.split(".")[0]
        return base.replace("-", " ").title()
    return "Unknown"


def guess_location(text: str) -> str:
    t = text or ""
    patterns = [
        r"\bBerlin\b",
        r"\bGermany\b",
        r"\bRemote\b",
        r"\bHybrid\b",
        r"\bMunich\b",
        r"\bHamburg\b",
    ]
    for p in patterns:
        m = re.search(p, t, re.I)
        if m:
            return m.group(0)
    return "Unknown"


def extract_present_skills(text: str, skills: Optional[List[str]] = None) -> List[str]:
    skills = skills or COMMON_SKILLS
    low = (text or "").lower()
    present = []
    for skill in skills:
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", low):
            present.append(skill)
    return sorted(set(present))


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> Dict[str, Any]:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return {"ok": True, "text": data.get("response", "").strip()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": "", "error": str(exc)}


def parse_contact_checks(cv_text: str) -> Dict[str, bool]:
    t = cv_text or ""
    return {
        "email": re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t) is not None,
        "phone": re.search(r"(\+?\d[\d\-\s()]{7,}\d)", t) is not None,
        "linkedin": re.search(r"(linkedin\.com|linkedin)", t, re.I) is not None,
        "github": re.search(r"(github\.com|github)", t, re.I) is not None,
        "location": re.search(r"\b(berlin|germany|remote|hybrid|munich|hamburg)\b", t, re.I) is not None,
    }
