from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from rag import cosine_similarity
from utils import COMMON_SKILLS, parse_contact_checks

IMPACT_WORDS = ["improved", "increased", "reduced", "optimized", "saved", "grew", "revenue", "cost", "%", "roi"]
HEADINGS = ["experience", "education", "skills", "projects"]


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "you",
    "your",
    "this",
    "that",
    "from",
    "are",
    "will",
    "have",
    "has",
    "role",
    "job",
    "required",
    "requirements",
    "candidate",
}


def extract_keywords(text: str, top_n: int = 35) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", (text or "").lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        if t in STOPWORDS:
            continue
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def compute_job_fit_score(
    profile_embedding: np.ndarray,
    job_embedding: np.ndarray,
    profile_skills: List[str],
    job_text: str,
) -> Dict:
    sim = max(0.0, cosine_similarity(profile_embedding, job_embedding))
    sim_score = min(70, int(round(sim * 70)))

    job_keywords = extract_keywords(job_text, top_n=35)
    profile_skill_set = {s.lower() for s in profile_skills}
    overlap = [kw for kw in job_keywords if kw in profile_skill_set]
    overlap_ratio = (len(overlap) / max(1, len(set(job_keywords))))
    overlap_score = min(30, int(round(overlap_ratio * 60)))

    total = min(100, sim_score + overlap_score)
    return {
        "total": int(total),
        "embedding_similarity": round(sim, 4),
        "embedding_score": sim_score,
        "keyword_overlap_score": overlap_score,
        "matched_profile_keywords": overlap[:20],
    }


def compute_missing_skills(job_text: str, cv_text: str) -> Dict:
    job_low = (job_text or "").lower()
    cv_low = (cv_text or "").lower()
    required = [s for s in COMMON_SKILLS if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", job_low)]
    missing = [s for s in required if not re.search(rf"(?<!\w){re.escape(s)}(?!\w)", cv_low)]
    matched = [s for s in required if s not in missing]
    return {"required": required, "missing": missing, "matched": matched}


def compute_ats_score(cv_text: str, job_text: str) -> Dict:
    cv = cv_text or ""
    cv_low = cv.lower()

    # Contact presence (20)
    contact = parse_contact_checks(cv)
    contact_score = 0
    contact_score += 4 if contact["email"] else 0
    contact_score += 4 if contact["phone"] else 0
    contact_score += 3 if contact["location"] else 0
    contact_score += 2 if contact["linkedin"] else 0
    contact_score += 2 if contact["github"] else 0
    contact_score = min(15, contact_score)

    # Standard headings (15)
    heading_hits = {h: bool(re.search(rf"\b{re.escape(h)}\b", cv_low)) for h in HEADINGS}
    section_score = 0
    section_score += 4 if heading_hits["experience"] else 0
    section_score += 4 if heading_hits["education"] else 0
    section_score += 4 if heading_hits["skills"] else 0
    section_score += 3 if heading_hits["projects"] else 0

    # Parsability (20)
    non_space = len(re.sub(r"\s+", "", cv))
    words = len(re.findall(r"[A-Za-z0-9]+", cv))
    parsability_score = 0
    parsability_score += 8 if non_space > 0 else 0
    parsability_score += 6 if words >= 120 else (3 if words >= 50 else 0)
    weird_ratio = len(re.findall(r"[^\w\s.,:%$€£()\-/]", cv)) / max(1, len(cv))
    parsability_score += 6 if weird_ratio < 0.05 else (3 if weird_ratio < 0.1 else 0)
    parsability_score = min(20, parsability_score)

    # Keyword match (30)
    jd_keywords = extract_keywords(job_text, top_n=40)
    matches = [kw for kw in jd_keywords if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", cv_low)]
    coverage = len(matches) / max(1, len(set(jd_keywords)))
    keyword_score = min(30, int(round(coverage * 30)))

    # Impact signals (20)
    impact_words = sum(1 for w in IMPACT_WORDS if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", cv_low))
    percents = len(re.findall(r"\d+\s?%", cv))
    currency = len(re.findall(r"[$€£]\s?\d", cv))
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", cv))
    impact_signal = impact_words + min(5, percents) + min(3, currency) + min(4, numbers // 8)
    impact_score = min(20, impact_signal * 2)

    total = min(100, contact_score + section_score + parsability_score + keyword_score + impact_score)
    return {
        "total": int(total),
        "contact_score": int(contact_score),
        "section_score": int(section_score),
        "parsability_score": int(parsability_score),
        "keyword_score": int(keyword_score),
        "impact_score": int(impact_score),
        "contact_checks": contact,
        "heading_hits": heading_hits,
        "keyword_matches": matches[:25],
    }
