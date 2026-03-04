from __future__ import annotations

import json
from typing import Dict, List

from pypdf import PdfReader

from prompts import profile_summary_prompt
from utils import call_ollama, extract_present_skills, normalize_whitespace


SECTION_HINTS = ["experience", "education", "skills", "projects", "summary"]


def extract_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    t = normalize_whitespace(text)
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


def heuristic_profile_summary(cv_text: str) -> Dict:
    lines = [ln.strip() for ln in cv_text.splitlines() if ln.strip()]
    top_lines = lines[:5]
    summary = " ".join(top_lines)[:280] or "Profile extracted from CV text."
    skills = extract_present_skills(cv_text)
    key_exp = []
    for ln in lines:
        low = ln.lower()
        if any(h in low for h in SECTION_HINTS):
            continue
        if len(ln) > 30:
            key_exp.append(ln)
        if len(key_exp) >= 3:
            break
    return {
        "summary": summary,
        "skills": skills,
        "key_experience": key_exp,
        "source": "heuristic",
    }


def summarize_profile(cv_text: str, model: str = "mistral:7b") -> Dict:
    prompt = profile_summary_prompt(cv_text)
    result = call_ollama(prompt=prompt, model=model, timeout=120)
    if not result.get("ok"):
        h = heuristic_profile_summary(cv_text)
        h["llm_error"] = result.get("error", "unknown")
        return h

    raw = result.get("text", "")
    # Try parse JSON. If not parseable, fallback.
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start : end + 1])
        return {
            "summary": parsed.get("summary", "").strip() or heuristic_profile_summary(cv_text)["summary"],
            "skills": parsed.get("skills", []) or extract_present_skills(cv_text),
            "key_experience": parsed.get("key_experience", [])[:3],
            "source": "ollama",
        }
    except Exception:  # noqa: BLE001
        h = heuristic_profile_summary(cv_text)
        h["llm_parse_warning"] = "Could not parse JSON response."
        return h


def build_profile_from_pages(pages: List[str], model: str = "mistral:7b") -> Dict:
    cv_text = "\n".join(pages).strip()
    chunks = []
    for i, page in enumerate(pages, start=1):
        for ci, ch in enumerate(chunk_text(page)):
            chunks.append({"page": i, "chunk_id": f"p{i}_c{ci}", "text": ch})

    profile_summary = summarize_profile(cv_text, model=model)
    return {
        "cv_text": cv_text,
        "pages": pages,
        "chunks": chunks,
        "profile_summary": profile_summary,
        "skills": profile_summary.get("skills", []),
    }
