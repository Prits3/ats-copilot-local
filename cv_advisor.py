from __future__ import annotations

import re
from typing import Dict, List

import requests

from profile import KNOWN_SKILLS

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:7b"
IMPACT_WORDS = ["improved", "increased", "reduced", "optimized", "saved", "grew", "revenue", "cost", "%", "roi"]
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
    "experience",
    "years",
    "year",
}


def _extract_required_skills(job_text: str) -> List[str]:
    low = (job_text or "").lower()
    req = [s for s in KNOWN_SKILLS if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", low)]
    return sorted(set(req))


def _extract_jd_keywords(job_text: str, top_n: int = 30) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", (job_text or "").lower())
    freq = {}
    for t in tokens:
        if t in STOPWORDS:
            continue
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def missing_skills(job_text: str, cv_text: str) -> Dict:
    req = _extract_required_skills(job_text)
    cv_low = (cv_text or "").lower()
    miss = [s for s in req if not re.search(rf"(?<!\w){re.escape(s)}(?!\w)", cv_low)]
    matched = [s for s in req if s not in miss]
    return {"required": req, "missing": miss, "matched": matched}


def ats_score(cv_text: str, job_text: str) -> Dict:
    cv = cv_text or ""
    cv_low = cv.lower()

    has_email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cv) is not None
    has_phone = re.search(r"(\+?\d[\d\-\s()]{7,}\d)", cv) is not None
    has_linkedin = re.search(r"(linkedin\.com|linkedin)", cv_low) is not None
    has_github = re.search(r"(github\.com|github)", cv_low) is not None
    has_location = re.search(r"\b(berlin|germany|remote|hybrid)\b", cv_low) is not None
    contact_score = (
        (4 if has_email else 0)
        + (4 if has_phone else 0)
        + (2 if has_linkedin else 0)
        + (2 if has_github else 0)
        + (3 if has_location else 0)
    )

    headings = {
        "experience": bool(re.search(r"\bexperience\b", cv_low)),
        "education": bool(re.search(r"\beducation\b", cv_low)),
        "skills": bool(re.search(r"\bskills\b", cv_low)),
        "projects": bool(re.search(r"\bprojects?\b", cv_low)),
    }
    section_score = (4 if headings["experience"] else 0) + (4 if headings["education"] else 0) + (4 if headings["skills"] else 0) + (3 if headings["projects"] else 0)

    non_space = len(re.sub(r"\s+", "", cv))
    words = len(re.findall(r"[A-Za-z0-9]+", cv))
    parse_score = 0
    parse_score += 8 if non_space > 0 else 0
    parse_score += 6 if words >= 120 else (3 if words >= 50 else 0)
    noise = len(re.findall(r"[^\w\s.,:%$€£()\-/]", cv)) / max(1, len(cv))
    parse_score += 6 if noise < 0.05 else (3 if noise < 0.1 else 0)
    parse_score = min(20, parse_score)

    req = _extract_required_skills(job_text)
    matched = [s for s in req if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", cv_low)]
    jd_keywords = _extract_jd_keywords(job_text)
    matched_jd_keywords = [kw for kw in jd_keywords if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", cv_low)]
    keyword_coverage = len(set(matched_jd_keywords)) / max(1, len(set(jd_keywords)))
    required_skill_component = len(matched) / max(1, len(req))
    keyword_score = min(30, int(round((0.6 * keyword_coverage + 0.4 * required_skill_component) * 30)))

    impact_words = sum(1 for w in IMPACT_WORDS if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", cv_low))
    percents = len(re.findall(r"\d+\s?%", cv))
    currency = len(re.findall(r"[$€£]\s?\d", cv))
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", cv))
    impact_signal = impact_words + min(4, percents) + min(3, currency) + min(4, numbers // 8)
    impact_score = min(20, impact_signal * 2)

    # Job-specific penalties for realism.
    penalty = 0
    if not has_linkedin:
        penalty += 3
    if not has_github:
        penalty += 2
    if not headings["skills"]:
        penalty += 6
    if percents == 0 and currency == 0 and numbers < 8:
        penalty += 6
    if words < 80:
        penalty += 8

    total = min(100, max(0, contact_score + section_score + parse_score + keyword_score + impact_score - penalty))
    return {
        "total": int(total),
        "contact_score": int(contact_score),
        "section_score": int(section_score),
        "parsability_score": int(parse_score),
        "keyword_score": int(keyword_score),
        "impact_score": int(impact_score),
        "penalty": int(penalty),
        "matched_keywords": matched,
        "matched_jd_keywords": matched_jd_keywords[:20],
        "required_keywords": req,
    }


def interpret_ats_score(ats: Dict) -> Dict:
    total = int(ats.get("total", 0))
    if total >= 85:
        band = "Strong"
        meaning = "Your CV is ATS-friendly for this job and has good keyword/structure alignment."
    elif total >= 70:
        band = "Good"
        meaning = "Decent ATS alignment, but there are clear improvements that can raise interview chances."
    elif total >= 55:
        band = "Needs Work"
        meaning = "ATS alignment is partial; important job-specific signals are missing."
    else:
        band = "Weak"
        meaning = "Low ATS fit for this job. CV structure and/or job keyword alignment need major improvement."

    tips: List[str] = []
    if ats.get("contact_score", 0) < 12:
        tips.append("Add complete contact info: email, phone, location, plus LinkedIn/GitHub links.")
    if ats.get("section_score", 0) < 12:
        tips.append("Use clear section headers: Experience, Education, Skills, Projects.")
    if ats.get("parsability_score", 0) < 14:
        tips.append("Improve parsability: keep plain text layout, avoid heavy graphics/tables, and ensure enough text content.")
    if ats.get("keyword_score", 0) < 18:
        tips.append("Increase job-specific keyword match by reflecting exact JD terms you truly have experience with.")
    if ats.get("impact_score", 0) < 10:
        tips.append("Add quantified impact to bullets (%, $, time saved, growth), only where truthful.")
    if ats.get("penalty", 0) > 0:
        tips.append("Reduce penalty factors: include Skills section, add profile links, and strengthen measurable outcomes.")

    if not tips:
        tips.append("Maintain this level by tailoring keywords per job and keeping achievements quantified.")

    return {
        "band": band,
        "meaning": meaning,
        "tips": tips,
    }


def build_why_match_and_action_plan(cv_evidence: List[Dict], missing: List[str]) -> Dict:
    why = []
    for ch in cv_evidence[:4]:
        txt = (ch.get("text", "") or "").replace("\n", " ").strip()
        if txt:
            why.append(f"(p{ch.get('page', '?')}) {txt[:170]}...")
    if not why:
        why = ["Limited CV evidence retrieved for this job match."]

    action = []
    for s in missing[:5]:
        action.append(f"Learn/practice `{s}` and add a truthful project bullet if you complete one.")
    if not action:
        action = ["Strengthen quantified impact in existing bullets (add %/$ only if true and evidenced)."]
    return {"why_match": why, "action_plan": action}


def detailed_ats_suggestions(ats: Dict, cv_text: str) -> List[Dict]:
    """Return per-category breakdown with specific, actionable fix text."""
    cv = cv_text or ""
    cv_low = cv.lower()

    # ── Contact (max 15) ──
    missing_contact = []
    if not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cv):
        missing_contact.append("email address")
    if not re.search(r"(\+?\d[\d\-\s()]{7,}\d)", cv):
        missing_contact.append("phone number")
    if "linkedin" not in cv_low:
        missing_contact.append("LinkedIn URL (linkedin.com/in/...)")
    if "github" not in cv_low:
        missing_contact.append("GitHub URL (github.com/...)")
    if not re.search(r"\b(berlin|germany|remote|hybrid)\b", cv_low):
        missing_contact.append("location (e.g. Berlin, Germany or Remote)")
    cs = ats.get("contact_score", 0)
    contact_fix = f"Missing: {', '.join(missing_contact)}" if missing_contact else "All contact fields present."

    # ── Sections (max 15) ──
    missing_sections = [s for s in ["Experience", "Education", "Skills", "Projects"] if s.lower() not in cv_low]
    ss = ats.get("section_score", 0)
    section_fix = (
        f"Add these section headers: {', '.join(missing_sections)}"
        if missing_sections else "All required sections present."
    )

    # ── Keywords (max 30) ──
    ks = ats.get("keyword_score", 0)
    required = ats.get("required_keywords", [])
    matched_req = ats.get("matched_keywords", [])
    missing_req = [s for s in required if s not in matched_req][:10]
    if missing_req:
        keyword_fix = (
            f"Add these {len(missing_req)} missing keywords (only if you genuinely have the skill): "
            + ", ".join(f"**{s}**" for s in missing_req)
        )
    elif ks < 22:
        keyword_fix = "Mirror the exact JD phrasing — use the same wording the JD uses for tools and methods."
    else:
        keyword_fix = "Strong keyword coverage. Mirror JD phrasing closely for edge cases."

    # ── Impact (max 20) ──
    impact_s = ats.get("impact_score", 0)
    percents = len(re.findall(r"\d+\s?%", cv))
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", cv))
    currency = len(re.findall(r"[$€£]\s?\d", cv))
    if impact_s < 8:
        impact_fix = (
            f"Only {percents} % figures and {numbers} numbers found. Add metrics to at least 4–5 bullets: "
            "e.g. 'Reduced query time by 40%', 'Processed 2M+ records daily', 'Grew DAU from 10K to 50K'."
        )
    elif impact_s < 14:
        impact_fix = (
            f"{percents} % figures, {currency} currency mentions found. "
            "Add 2–3 more: include time saved, revenue impact, or scale (users, records, cost)."
        )
    else:
        impact_fix = f"Strong metrics: {percents} % figures, {numbers} numbers, {currency} currency mentions."

    return [
        {
            "category": "Contact Info", "score": cs, "max": 15,
            "status": "good" if cs >= 12 else ("warn" if cs >= 8 else "bad"),
            "fix": contact_fix,
        },
        {
            "category": "Section Headers", "score": ss, "max": 15,
            "status": "good" if ss >= 12 else ("warn" if ss >= 8 else "bad"),
            "fix": section_fix,
        },
        {
            "category": "Keyword Match", "score": ks, "max": 30,
            "status": "good" if ks >= 22 else ("warn" if ks >= 14 else "bad"),
            "fix": keyword_fix,
        },
        {
            "category": "Impact & Metrics", "score": impact_s, "max": 20,
            "status": "good" if impact_s >= 14 else ("warn" if impact_s >= 8 else "bad"),
            "fix": impact_fix,
        },
    ]


def cv_improvement_suggestions(top_jobs: List[Dict], cv_text: str) -> List[str]:
    tips = []
    seen_missing = set()
    for job in top_jobs[:3]:
        ms = missing_skills(job.get("description_text", ""), cv_text)["missing"]
        for m in ms:
            seen_missing.add(m)
    if seen_missing:
        tips.append(
            "Add missing keywords only when truthful and supported by your real work: "
            + ", ".join(sorted(seen_missing))
        )
    tips.append("Reorder Skills and top Experience bullets to mirror target job wording.")
    tips.append("Add quantified impact prompts where possible: e.g., 'improved X by Y%'.")
    return tips


def tailor_cv_with_ollama(job_text: str, evidence: List[Dict], model: str = DEFAULT_MODEL) -> Dict:
    evidence_md = "\n".join(
        [f"- (p{e.get('page', '?')}) {e.get('text', '').replace(chr(10), ' ')[:280]}" for e in evidence[:8]]
    )
    prompt = f"""
You are a strict CV tailoring assistant.
Use only provided evidence. No fabrication.

Return markdown with:
1) Recommended summary statement (1-2 lines)
2) Skills reorder + suggested keywords (mark missing if not evidenced)
3) Rewrite up to 3 bullets in STAR style (only use metrics if shown in evidence)
4) Missing skills

JOB:
{job_text[:10000]}

CV EVIDENCE:
{evidence_md}
""".strip()
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return {"ok": True, "text": r.json().get("response", "").strip()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": "", "error": str(exc)}
