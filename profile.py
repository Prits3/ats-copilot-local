from __future__ import annotations

import os
import re
import tempfile
from io import BytesIO
from typing import Dict, List

import requests
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:7b"

KNOWN_SKILLS = [
    "python",
    "sql",
    "power bi",
    "tableau",
    "excel",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "machine learning",
    "ml",
    "risk",
    "finance",
    "product analytics",
    "a/b testing",
    "statistics",
    "dbt",
    "airflow",
    "aws",
    "gcp",
    "azure",
]


def extract_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages


def extract_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages


def uploaded_pdf_to_pages(uploaded_file) -> List[str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        return extract_pdf_pages(path)
    finally:
        os.unlink(path)


def redact_sensitive_text(text: str) -> str:
    t = text or ""
    # Remove lines that start with sensitive labels.
    sensitive_labels = [
        "Nationality",
        "Date of birth",
        "DOB",
        "Gender",
        "Marital status",
    ]
    lines = []
    for ln in t.splitlines():
        low = ln.strip().lower()
        if any(low.startswith(lbl.lower()) for lbl in sensitive_labels):
            continue
        lines.append(ln)
    t = "\n".join(lines)
    # Redact common DOB date formats.
    t = re.sub(r"\b([0-3]?\d[./][0-1]?\d[./](?:19|20)\d{2})\b", "[REDACTED_DOB]", t)
    return t


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    chunks = []
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        chunks.append(t[start:end])
        if end == len(t):
            break
        start = end - overlap
    return chunks


def extract_profile_signals(cv_text: str) -> Dict:
    text_low = cv_text.lower()
    skills = []
    for s in KNOWN_SKILLS:
        if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", text_low):
            skills.append(s)

    impact_lines = []
    for ln in cv_text.splitlines():
        line = ln.strip()
        if not line:
            continue
        if re.search(r"(\d+\s?%|[$€£]\s?\d+|\b(increased|improved|saved|reduced|grew|optimized)\b)", line, re.I):
            impact_lines.append(line)
        if len(impact_lines) >= 8:
            break

    return {
        "skills": sorted(set(skills)),
        "impact_lines": impact_lines,
    }


def build_profile_summary_with_ollama(cv_text: str, model: str = DEFAULT_MODEL) -> Dict:
    prompt = f"""
You are a strict CV parser. Use only the CV text below.
Return plain text with sections:
SUMMARY:
SKILLS:
EXPERIENCE_SIGNALS:
Do not invent information.

CV:
{cv_text[:12000]}
""".strip()
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return {"ok": True, "text": r.json().get("response", "").strip()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "text": ""}


def build_profile(pages: List[str], model: str = DEFAULT_MODEL) -> Dict:
    redacted_pages = [redact_sensitive_text(p) for p in pages]
    cv_text = "\n".join(redacted_pages).strip()
    chunks = []
    for pi, p in enumerate(redacted_pages, start=1):
        for ci, ch in enumerate(chunk_text(p)):
            chunks.append({"page": pi, "chunk_id": f"p{pi}_c{ci}", "text": ch})

    signals = extract_profile_signals(cv_text)
    llm_summary = build_profile_summary_with_ollama(cv_text, model=model)

    return {
        "cv_text": cv_text,
        "pages": redacted_pages,
        "chunks": chunks,
        "skills": signals["skills"],
        "impact_lines": signals["impact_lines"],
        "llm_summary": llm_summary,
    }
