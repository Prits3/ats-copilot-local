"""Bullet Rewriter — transforms weak CV bullets into impact-driven, role-specific ones."""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

WEAK_PHRASES = [
    "responsible for", "worked on", "helped with", "assisted with",
    "involved in", "participated in", "was part of", "supported",
    "contributed to", "had experience", "exposure to", "tasked with",
    "duties included", "worked closely", "helped to",
]

STRONG_VERBS: dict[str, List[str]] = {
    "data_analyst": [
        "Analyzed", "Built", "Designed", "Automated", "Reduced", "Improved",
        "Generated", "Visualized", "Delivered", "Tracked", "Optimized", "Identified",
    ],
    "business_analyst": [
        "Defined", "Mapped", "Streamlined", "Identified", "Documented", "Aligned",
        "Facilitated", "Delivered", "Modeled", "Reduced", "Improved", "Translated",
    ],
    "data_scientist": [
        "Developed", "Built", "Trained", "Deployed", "Implemented", "Optimized",
        "Reduced", "Improved", "Predicted", "Modeled", "Engineered", "Evaluated",
    ],
    "ml_engineer": [
        "Deployed", "Architected", "Built", "Optimized", "Reduced", "Scaled",
        "Integrated", "Automated", "Implemented", "Engineered", "Monitored",
    ],
    "software_engineer": [
        "Built", "Developed", "Architected", "Optimized", "Refactored", "Deployed",
        "Integrated", "Automated", "Implemented", "Shipped", "Scaled", "Led",
    ],
    "product_manager": [
        "Led", "Launched", "Defined", "Drove", "Prioritized", "Shipped", "Delivered",
        "Collaborated", "Managed", "Scaled", "Aligned", "Reduced", "Grew",
    ],
    "default": [
        "Led", "Built", "Delivered", "Improved", "Developed", "Implemented",
        "Optimized", "Reduced", "Increased", "Automated", "Launched", "Designed",
    ],
}


def has_weak_language(bullet: str) -> bool:
    lower = bullet.lower()
    return any(weak in lower for weak in WEAK_PHRASES)


def has_impact_metric(bullet: str) -> bool:
    return bool(re.search(
        r"\d+\s*(%|x|×|k|K|\$|€|£|hours?|days?|weeks?|months?|users?|customers?|percent)",
        bullet, re.IGNORECASE,
    ))


def clean_bullet(text: str) -> str:
    """Strip leading bullet symbols and whitespace."""
    return re.sub(r"^[\-\•\*\d+\.\)]+\s*", "", text.strip())


def rewrite_bullet(
    bullet: str,
    role_type: str,
    jd_text: str,
    llm_fn: Callable[[str], str],
) -> str:
    """Rewrite a single bullet for impact and role alignment."""
    prompt = f"""Rewrite this CV bullet point for a {role_type.replace("_", " ")} role.

Original: {bullet}

Job context (key skills/requirements): {jd_text[:400]}

Rules:
1. Start with a strong action verb — NOT "Responsible for", "Worked on", "Helped", "Assisted"
2. Include or estimate measurable impact (numbers, %, scale, time saved, revenue)
3. Keep it under 20 words — concise and specific
4. Align wording with what matters for a {role_type.replace("_", " ")} role
5. Remove all generic filler language

Return ONLY the rewritten bullet, no explanation, no bullet symbol."""

    result = llm_fn(prompt).strip()
    result = clean_bullet(result)
    # Reject if LLM returned something useless
    if not result or len(result) < 10:
        return bullet
    return result


def rewrite_experience_bullets(
    experience: dict,
    jd_analysis: dict,
    jd_text: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """Return a copy of the experience with rewritten bullets."""
    rewritten = dict(experience)
    role_type = jd_analysis.get("role_type", "default")

    if llm_fn is None:
        return rewritten

    new_bullets = []
    for bullet in experience.get("bullets", []):
        try:
            new_bullet = rewrite_bullet(bullet, role_type, jd_text, llm_fn)
            new_bullets.append(new_bullet)
        except Exception:
            new_bullets.append(bullet)

    rewritten["bullets"] = new_bullets
    return rewritten


def generate_positioning_summary(
    profile: dict,
    jd_analysis: dict,
    positioning: str,
    llm_fn: Callable[[str], str],
) -> str:
    """Generate a tailored 2-3 sentence professional summary."""
    role_label = jd_analysis.get("role_type", "analyst").replace("_", " ")
    skills_str = ", ".join(profile.get("skills", {}).get("technical", [])[:8])
    jd_skills_str = ", ".join(jd_analysis.get("required_skills", [])[:6])
    company_type = jd_analysis.get("company_type", "startup")
    seniority = jd_analysis.get("seniority", "junior")

    # Pull real experience context for a grounded summary
    experiences = profile.get("roles", profile.get("experiences", []))[:3]
    exp_context = "; ".join(
        f"{e.get('title','')} at {e.get('company','')}" for e in experiences if e.get("title")
    )
    projects = profile.get("projects", [])[:2]
    proj_context = "; ".join(p.get("name", "") for p in projects if p.get("name"))

    prompt = f"""You are writing the PROFILE section of a professional CV.

Write 2-3 tight, honest sentences that position this candidate for the target role.

Candidate background:
- Experiences: {exp_context or 'see below'}
- Projects: {proj_context or 'see below'}
- Key skills: {skills_str}
- Positioning: {positioning}

Target role: {seniority} {role_label} at a {company_type}
Role requires: {jd_skills_str}

Rules:
1. Grounded in REAL experience above — no fabrication
2. Does NOT start with "I" — write in first person but starting with a noun or verb phrase
3. Mentions 1-2 specific things the candidate has actually done
4. Ends with what the candidate is looking for / value they bring to this role
5. No clichés: passionate, driven, dynamic, results-oriented, team player, hard-working
6. 50-70 words max — read like a confident, direct human wrote it

Return ONLY the summary paragraph, no explanation, no quotes."""

    result = llm_fn(prompt).strip()
    return result if len(result) > 20 else ""


def holistic_cv_rewrite(
    profile: dict,
    selected_experiences: List[dict],
    selected_projects: List[dict],
    jd_text: str,
    jd_analysis: dict,
    positioning: str,
    llm_fn: Callable[[str], str],
) -> Dict:
    """Single-shot LLM rewrite of the entire CV at once.

    Returns a dict with keys:
      summary  — rewritten profile paragraph
      experiences — list of experience dicts with rewritten bullets
      projects    — list of project dicts with rewritten bullets
    Falls back to original data on any error.
    """
    role_label  = jd_analysis.get("role_type", "default").replace("_", " ")
    seniority   = jd_analysis.get("seniority", "mid")
    req_skills  = ", ".join(jd_analysis.get("required_skills", [])[:8])
    company_type = jd_analysis.get("company_type", "startup")
    responsibilities = "; ".join(jd_analysis.get("key_responsibilities", [])[:4])

    # Build compact experience/project summaries for the prompt
    exp_lines = []
    for i, exp in enumerate(selected_experiences):
        bullets_str = "\n".join(f"    - {b}" for b in exp.get("bullets", [])[:5])
        exp_lines.append(
            f"[EXP_{i}] {exp.get('title','')} at {exp.get('company','')} "
            f"({exp.get('start_date','')} – {exp.get('end_date','Present')}):\n{bullets_str}"
        )

    proj_lines = []
    for i, proj in enumerate(selected_projects):
        bullets_str = "\n".join(f"    - {b}" for b in proj.get("bullets", [])[:3])
        proj_lines.append(f"[PROJ_{i}] {proj.get('name','')}: {proj.get('description','')}\n{bullets_str}")

    skills_raw = profile.get("skills", {})
    all_skills = (skills_raw.get("technical", []) + skills_raw.get("tools", []))[:14]

    prompt = f"""You are an elite CV writer. Rewrite this candidate's CV for the target role in ONE shot.

TARGET ROLE: {seniority} {role_label} at a {company_type}
POSITIONING: {positioning}
JOB REQUIRES: {req_skills}
KEY RESPONSIBILITIES: {responsibilities}

CANDIDATE SKILLS: {', '.join(all_skills)}

CANDIDATE EXPERIENCES:
{chr(10).join(exp_lines)}

CANDIDATE PROJECTS:
{chr(10).join(proj_lines)}

JOB DESCRIPTION (first 1200 chars):
{jd_text[:1200]}

REWRITE RULES:
1. Each bullet: strong action verb + specific outcome/metric (use real numbers from original where available; do NOT invent metrics)
2. Max 4 bullets per experience, 2 bullets per project
3. Each bullet under 25 words — sharp and specific
4. Remove all weak language: "responsible for", "worked on", "helped", "involved in"
5. Tailor wording to match job description keywords naturally
6. Summary: 2-3 tight sentences, no "I", no clichés, highlight overlap between candidate and role

Return ONLY valid JSON matching this exact schema:
{{
  "summary": "2-3 sentence professional summary tailored to the role",
  "experiences": [
    {{"id": "EXP_0", "bullets": ["bullet 1", "bullet 2", "bullet 3"]}}
  ],
  "projects": [
    {{"id": "PROJ_0", "bullets": ["bullet 1", "bullet 2"]}}
  ]
}}"""

    raw = llm_fn(prompt).strip()

    # Parse response
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return _fallback_result(selected_experiences, selected_projects)
    except (json.JSONDecodeError, AttributeError):
        return _fallback_result(selected_experiences, selected_projects)

    # Map rewritten bullets back onto experience/project dicts
    exp_map: Dict[str, List[str]] = {e["id"]: e["bullets"] for e in data.get("experiences", [])}
    proj_map: Dict[str, List[str]] = {p["id"]: p["bullets"] for p in data.get("projects", [])}

    rewritten_exps = []
    for i, exp in enumerate(selected_experiences):
        new_exp = dict(exp)
        new_bullets = exp_map.get(f"EXP_{i}", exp.get("bullets", []))
        new_exp["bullets"] = [b for b in new_bullets if b.strip()][:4] or exp.get("bullets", [])
        rewritten_exps.append(new_exp)

    rewritten_projs = []
    for i, proj in enumerate(selected_projects):
        new_proj = dict(proj)
        new_bullets = proj_map.get(f"PROJ_{i}", proj.get("bullets", []))
        new_proj["bullets"] = [b for b in new_bullets if b.strip()][:2] or proj.get("bullets", [])
        rewritten_projs.append(new_proj)

    return {
        "summary": data.get("summary", ""),
        "experiences": rewritten_exps,
        "projects": rewritten_projs,
    }


def _fallback_result(selected_experiences: List[dict], selected_projects: List[dict]) -> Dict:
    return {
        "summary": "",
        "experiences": selected_experiences,
        "projects": selected_projects,
    }
