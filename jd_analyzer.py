"""Job Description Analyzer — extracts structured signal from raw JD text."""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

ROLE_TYPES = [
    "data_analyst", "business_analyst", "data_scientist", "ml_engineer",
    "software_engineer", "product_manager", "operations", "finance", "marketing",
]

_ROLE_KEYWORDS: dict[str, list[str]] = {
    "data_analyst": ["data analyst", "analytics", "bi analyst", "reporting analyst", "power bi", "tableau", "looker"],
    "business_analyst": ["business analyst", "requirements", "process improvement", "stakeholder", "user stories"],
    "data_scientist": ["data scientist", "machine learning", "statistical model", "mlops", "deep learning"],
    "ml_engineer": ["ml engineer", "mlops", "model deployment", "inference", "model serving", "model pipeline"],
    "software_engineer": ["software engineer", "developer", "backend", "frontend", "full stack", "swe", "sde"],
    "product_manager": ["product manager", "product owner", "roadmap", "go-to-market", "product strategy"],
    "operations": ["operations manager", "ops", "supply chain", "logistics", "process manager"],
    "finance": ["finance analyst", "fp&a", "controller", "treasury", "financial planning"],
    "marketing": ["marketing", "growth", "seo", "content", "campaign", "brand"],
}

_SKILL_PATTERNS = [
    "python", "sql", "r", "java", "scala", "go", "javascript", "typescript",
    "spark", "hadoop", "kafka", "airflow", "dbt", "flink",
    "power bi", "tableau", "looker", "excel", "qlik",
    "aws", "gcp", "azure", "databricks", "snowflake", "bigquery", "redshift",
    "docker", "kubernetes", "terraform", "git", "ci/cd",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "huggingface",
    "fastapi", "flask", "django", "react", "node",
    "jira", "confluence", "notion", "figma",
    "mlflow", "sagemaker", "vertex ai",
    "a/b testing", "statistics", "machine learning", "deep learning",
    "nlp", "computer vision", "llm", "rag",
]


def analyze_jd(jd_text: str, llm_fn: Optional[Callable[[str], str]] = None) -> dict:
    """Extract structured data from job description. Uses LLM if available."""
    if llm_fn:
        result = _analyze_with_llm(jd_text, llm_fn)
        if result:
            return result
    return _analyze_fallback(jd_text)


def _analyze_with_llm(jd_text: str, llm_fn: Callable[[str], str]) -> Optional[dict]:
    prompt = f"""Analyze this job description and return a JSON object.

Job Description:
{jd_text[:3000]}

Return ONLY a valid JSON object with exactly these fields:
{{
  "role_type": "one of: data_analyst, business_analyst, data_scientist, ml_engineer, software_engineer, product_manager, operations, finance, marketing",
  "seniority": "one of: intern, junior, mid, senior, lead",
  "required_skills": ["list of up to 12 technical skills explicitly mentioned"],
  "nice_to_have_skills": ["list of up to 5 bonus skills"],
  "keywords": ["top 10 most important ATS keywords from this JD"],
  "tone": "one of: analytical, execution, strategic, technical, creative",
  "key_responsibilities": ["top 5 responsibilities as short action phrases"],
  "success_criteria": "what success looks like in this role (1 sentence)",
  "company_type": "one of: startup, scaleup, enterprise, agency, consultancy",
  "industry": "the industry sector"
}}

Return ONLY the JSON, no markdown, no explanation."""

    raw = llm_fn(prompt)
    if not raw:
        return None
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _analyze_fallback(jd_text: str) -> dict:
    """Deterministic keyword-based fallback when LLM is unavailable."""
    text_lower = jd_text.lower()

    role_type = "data_analyst"
    for rt, kws in _ROLE_KEYWORDS.items():
        if any(kw in text_lower for kw in kws):
            role_type = rt
            break

    seniority = "mid"
    if any(w in text_lower for w in ["intern", "internship", "werkstudent", "working student"]):
        seniority = "intern"
    elif any(w in text_lower for w in ["junior", "entry level", "entry-level", "graduate"]):
        seniority = "junior"
    elif any(w in text_lower for w in ["senior", "lead", "principal", "staff"]):
        seniority = "senior"

    required_skills = [s for s in _SKILL_PATTERNS if s in text_lower][:12]

    company_type = "startup"
    if any(w in text_lower for w in ["series a", "series b", "seed", "early stage"]):
        company_type = "startup"
    elif any(w in text_lower for w in ["series c", "series d", "unicorn", "growth stage"]):
        company_type = "scaleup"
    elif any(w in text_lower for w in ["fortune", "global", "enterprise", "corporation"]):
        company_type = "enterprise"

    return {
        "role_type": role_type,
        "seniority": seniority,
        "required_skills": required_skills,
        "nice_to_have_skills": [],
        "keywords": required_skills[:10],
        "tone": "analytical",
        "key_responsibilities": [],
        "success_criteria": "",
        "company_type": company_type,
        "industry": "technology",
    }


def extract_master_profile_from_cv(cv_text: str, llm_fn: Callable[[str], str]) -> Optional[dict]:
    """Use LLM to structure raw CV text into master profile JSON format."""
    prompt = f"""You are a CV parser. Extract structured data from this CV text.

CRITICAL RULES — read carefully before extracting:
- IGNORE any "Why [Company]" or "Why I want to work here" sections — they are cover letter content
- IGNORE any "Leadership & Community" or "Volunteer" roles — leave out of roles list
- IGNORE any application-specific content — only extract the standard CV sections
- roles.bullets: max 4 bullets per role, keep the most impactful ones
- projects.bullets: max 2 bullets per project
- achievements: ONLY real awards, competitions, publications — NOT job duties, NOT cover letter text
- certifications: ONLY actual certificates with a name — NOT long sentences
- Each bullet must be a single short sentence (under 25 words) — NOT a full paragraph

CV Text:
{cv_text[:6000]}

Return ONLY a valid JSON object with exactly this structure (no extra fields):
{{
  "personal": {{
    "name": "full name",
    "tagline": "professional headline/subtitle line under the name if present",
    "email": "email",
    "phone": "phone",
    "linkedin": "linkedin url or profile handle",
    "github": "github url or username",
    "location": "city, country"
  }},
  "roles": [
    {{
      "id": "role_1",
      "title": "job title",
      "company": "company name",
      "start_date": "Mon YYYY",
      "end_date": "Mon YYYY or Present",
      "location": "city",
      "bullets": ["max 4 bullets, each under 25 words"],
      "skills": ["skill1", "skill2"],
      "tags": ["data_analyst", "software_engineer", etc]
    }}
  ],
  "projects": [
    {{
      "id": "proj_1",
      "name": "project name",
      "description": "one-line description",
      "bullets": ["max 2 bullets"],
      "skills": ["skill1", "skill2"],
      "tags": [],
      "url": "url if present"
    }}
  ],
  "education": [
    {{
      "institution": "university name",
      "degree": "BSc/MSc/etc",
      "field": "field of study",
      "start_date": "YYYY",
      "end_date": "YYYY or Present",
      "gpa": "grade if present",
      "location": "city if present",
      "relevant_courses": []
    }}
  ],
  "skills": {{
    "technical": ["up to 12 technical skills"],
    "tools": ["up to 8 tools/software"],
    "soft": ["up to 4 soft skills"],
    "languages": ["human languages if listed, e.g. English C2"]
  }},
  "achievements": ["ONLY real awards or publications — leave empty list if none"],
  "certifications": ["ONLY certificate names — leave empty list if none"]
}}

Return ONLY valid JSON. No markdown, no explanation."""

    raw = llm_fn(prompt)
    if not raw:
        return None
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            extracted = json.loads(match.group())
            return sanitize_profile(extracted)
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


# ── Keywords that flag non-CV application-specific content ───────────────────
_JUNK_KEYWORDS = {
    "why ", "clarity-first", "motivated by", "i'm drawn", "i already use",
    "0→1", "looking to grow", "i want to", "i am drawn", "ambiguous environment",
    "predefined process", "fast-paced environment",
}


def sanitize_profile(profile: dict) -> dict:
    """Strip junk from a raw extracted profile: WHY sections, over-long bullets, empty roles."""
    if not isinstance(profile, dict):
        return profile

    def _is_real_item(s: str, max_len: int = 120) -> bool:
        if not isinstance(s, str) or len(s.strip()) < 3:
            return False
        if len(s) > max_len:  # paragraphs are not cert names
            return False
        low = s.lower()
        return not any(kw in low for kw in _JUNK_KEYWORDS)

    def _is_real_bullet(b: str) -> bool:
        if not isinstance(b, str) or len(b.strip()) < 8:
            return False
        low = b.lower()
        return not any(kw in low for kw in _JUNK_KEYWORDS)

    # Wipe application-specific content
    profile["achievements"] = [a for a in profile.get("achievements", []) if _is_real_item(a)]
    profile["certifications"] = [c for c in profile.get("certifications", []) if _is_real_item(c)]

    # Cap + clean bullets per role (max 5 stored; renderer caps display at 4)
    for role in profile.get("roles", []):
        role["bullets"] = [b for b in role.get("bullets", []) if _is_real_bullet(b)][:5]

    # Cap + clean bullets per project (max 3 stored; renderer caps display at 2)
    for proj in profile.get("projects", []):
        proj["bullets"] = [b for b in proj.get("bullets", []) if _is_real_bullet(b)][:3]

    # Drop completely empty roles
    profile["roles"] = [r for r in profile.get("roles", []) if r.get("title") or r.get("company")]

    # Drop garbage roles: generic fallback title with no real company
    def _is_garbage_role(role: dict) -> bool:
        title = (role.get("title") or "").strip().lower()
        company = (role.get("company") or "").strip()
        if title in {"professional experience", "work experience", "experience"} and not company:
            return True
        # Role where bullets contain email addresses (contact info leaked in)
        bullets = role.get("bullets", [])
        if bullets and all("@" in b or re.match(r"^[\+\d]", b) for b in bullets):
            return True
        return False

    profile["roles"] = [r for r in profile.get("roles", []) if not _is_garbage_role(r)]

    # Strip bullets that are clearly contact info
    for role in profile.get("roles", []):
        role["bullets"] = [
            b for b in role.get("bullets", [])
            if "@" not in b and not re.match(r"^[\+\d\(\)\s\-\.]{7,}$", b)
        ]

    return profile
