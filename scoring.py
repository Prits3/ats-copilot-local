from typing import List, Dict
import re

IMPACT_WORDS = ["improved", "increased", "reduced", "optimized", "saved", "grew", "%", "roi", "revenue", "cost"]

SKILLS = [
    "python","sql","excel","power bi","tableau","pandas","numpy","scikit-learn","pytorch","tensorflow",
    "nlp","llm","rag","fastapi","docker","git","aws","gcp","azure","kubernetes","mlops"
]

def simple_skill_list(text: str) -> List[str]:
    t = text.lower()
    return [s for s in SKILLS if s in t]

def score_candidate(jd_text: str, retrieved: List[Dict]) -> Dict:
    jd_skills = set(simple_skill_list(jd_text))
    evidence_text = " ".join([r["text"] for r in retrieved]).lower()
    cv_skills = set(simple_skill_list(evidence_text))

    overlap = len(jd_skills.intersection(cv_skills))
    skills_score = int(40 * (overlap / max(1, len(jd_skills)))) if jd_skills else 0

    rel = (sum(r["score"] for r in retrieved) / len(retrieved)) if retrieved else 0.0
    relevance_score = int(max(0, min(30, rel * 30)))

    jd_low = jd_text.lower()
    seniority_score = 8
    if "junior" in jd_low or "entry" in jd_low or "graduate" in jd_low:
        seniority_score = 12

    impact_hits = sum(1 for w in IMPACT_WORDS if w in evidence_text)
    impact_score = min(10, impact_hits)

    must = 0
    if "python" in jd_low:
        must += 3 if "python" in evidence_text else 0
    if "sql" in jd_low:
        must += 2 if "sql" in evidence_text else 0

    total = min(100, skills_score + relevance_score + seniority_score + impact_score + must)

    return {
        "total": total,
        "skills_score": skills_score,
        "relevance_score": relevance_score,
        "seniority_score": seniority_score,
        "impact_score": impact_score,
        "must_have_score": must,
        "jd_skills": sorted(jd_skills),
        "cv_skills": sorted(cv_skills),
        "overlap": overlap,
    }


def _skill_present(text: str, skill: str) -> bool:
    pattern = rf"(?<!\w){re.escape(skill.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def _skill_evidence_lines(cv_text: str, skill: str, max_lines: int = 3) -> List[str]:
    lines = [ln.strip() for ln in cv_text.splitlines() if ln.strip()]
    out: List[str] = []
    for line in lines:
        if _skill_present(line, skill):
            out.append(line)
        if len(out) >= max_lines:
            break
    return out


def score_skill_match(jd_text: str, cv_text: str) -> Dict:
    jd_skills = [s for s in SKILLS if _skill_present(jd_text, s)]
    matched = [s for s in jd_skills if _skill_present(cv_text, s)]
    missing = [s for s in jd_skills if s not in matched]
    evidence = {s: _skill_evidence_lines(cv_text, s) for s in matched}
    return {
        "required_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_count": len(matched),
        "missing_count": len(missing),
        "evidence": evidence,
    }


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "to", "with", "you", "your", "we", "our", "will", "this",
    "role", "job", "required", "requirements", "candidate", "experience", "years", "year",
}

SECTION_PATTERNS = {
    "experience": re.compile(r"\b(experience|work experience|employment)\b", re.I),
    "education": re.compile(r"\b(education|academic|qualification)\b", re.I),
    "skills": re.compile(r"\b(skills|technical skills|core skills)\b", re.I),
    "projects": re.compile(r"\b(projects|project experience)\b", re.I),
}


def _extract_jd_keywords(jd_text: str, top_n: int = 25) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", jd_text.lower())
    clean = [t for t in tokens if t not in STOPWORDS]
    freq = {}
    for t in clean:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def _safe_ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_ats_score(cv_text: str, jd_text: str) -> Dict:
    cv = cv_text or ""
    cv_low = cv.lower()

    # 1) Contact & basics present (0-15)
    contact_score = 0
    has_email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cv) is not None
    has_phone = re.search(r"(\+?\d[\d\-\s()]{7,}\d)", cv) is not None
    has_location = re.search(
        r"\b(location|based in|remote|relocate|[A-Z][a-z]+,\s*[A-Z]{2}|[A-Z][a-z]+,\s*[A-Z][a-z]+)\b",
        cv,
    ) is not None
    has_linkedin = re.search(r"(linkedin\.com|linkedin)", cv_low) is not None
    has_github = re.search(r"(github\.com|github)", cv_low) is not None

    contact_score += 4 if has_email else 0
    contact_score += 4 if has_phone else 0
    contact_score += 3 if has_location else 0
    contact_score += 2 if has_linkedin else 0
    contact_score += 2 if has_github else 0

    # 2) Standard section headings (0-15)
    section_hits = {name: bool(p.search(cv)) for name, p in SECTION_PATTERNS.items()}
    section_score = (
        (4 if section_hits["experience"] else 0)
        + (4 if section_hits["education"] else 0)
        + (4 if section_hits["skills"] else 0)
        + (3 if section_hits["projects"] else 0)
    )

    # 3) Keyword density / match (0-30)
    jd_keywords = _extract_jd_keywords(jd_text)
    matched_keywords = [kw for kw in jd_keywords if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", cv_low)]
    coverage = _safe_ratio(len(matched_keywords), len(jd_keywords))
    cv_tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", cv_low)
    cv_len = len(cv_tokens)
    match_occ = 0
    if jd_keywords:
        for kw in jd_keywords:
            match_occ += len(re.findall(rf"(?<!\w){re.escape(kw)}(?!\w)", cv_low))
    density = _safe_ratio(match_occ, cv_len)
    spam_penalty = 0
    if density > 0.12:
        spam_penalty = 6
    elif density > 0.08:
        spam_penalty = 3
    keyword_score = max(0, min(30, int(round(coverage * 30)) - spam_penalty))

    # 4) Readability / parsing (0-20)
    non_space = len(re.sub(r"\s+", "", cv))
    word_count = len(cv_tokens)
    alpha_chars = sum(ch.isalpha() for ch in cv)
    alpha_ratio = _safe_ratio(alpha_chars, max(1, non_space))
    readability_score = 0
    readability_score += 8 if non_space > 0 else 0
    readability_score += 6 if word_count >= 120 else (3 if word_count >= 40 else 0)
    readability_score += 6 if 0.55 <= alpha_ratio <= 0.95 else (3 if 0.45 <= alpha_ratio <= 0.98 else 0)
    readability_score = min(20, readability_score)

    # 5) Quantified impact (0-20)
    impact_word_hits = sum(1 for w in IMPACT_WORDS if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", cv_low))
    number_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", cv))
    money_hits = len(re.findall(r"[$€£]\s?\d", cv))
    percent_hits = len(re.findall(r"\d+\s?%", cv))
    impact_signal = impact_word_hits + min(5, number_hits // 5) + min(3, money_hits) + min(4, percent_hits)
    impact_score = min(20, impact_signal * 2)

    total = min(100, contact_score + section_score + keyword_score + readability_score + impact_score)

    return {
        "total": int(total),
        "contact_score": int(contact_score),
        "section_score": int(section_score),
        "keyword_score": int(keyword_score),
        "readability_score": int(readability_score),
        "impact_score": int(impact_score),
        "matched_keyword_count": len(matched_keywords),
        "jd_keyword_count": len(jd_keywords),
        "matched_keywords": matched_keywords[:20],
        "section_hits": section_hits,
        "contact_checks": {
            "email": has_email,
            "phone": has_phone,
            "location": has_location,
            "linkedin": has_linkedin,
            "github": has_github,
        },
    }
