"""Relevance Engine — scores and selects experiences/projects against a job description."""
from __future__ import annotations

from typing import List

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _exp_to_text(exp: dict) -> str:
    parts = [
        f"{exp.get('title', '')} at {exp.get('company', '')}",
        *exp.get("bullets", []),
        "Skills: " + ", ".join(exp.get("skills", [])),
        "Tags: " + ", ".join(exp.get("tags", [])),
    ]
    return " ".join(parts)


def _proj_to_text(proj: dict) -> str:
    parts = [
        proj.get("name", ""),
        proj.get("description", ""),
        *proj.get("bullets", []),
        "Skills: " + ", ".join(proj.get("skills", [])),
    ]
    return " ".join(parts)


def _skill_overlap_score(item_skills: list, jd_skills: list) -> float:
    if not jd_skills:
        return 0.0
    item_set = {s.lower() for s in item_skills}
    jd_set = {s.lower() for s in jd_skills}
    return len(item_set & jd_set) / len(jd_set)


def rank_experiences(
    profile: dict,
    jd_text: str,
    jd_analysis: dict,
    embedder,
    top_n: int = 4,
) -> List[dict]:
    """Return top_n most relevant experiences, scored by semantic + skill overlap."""
    roles = profile.get("roles", [])
    if not roles:
        return []

    jd_emb = embedder.embed([jd_text[:2000]])[0]
    jd_skills = jd_analysis.get("required_skills", []) + jd_analysis.get("keywords", [])

    scored = []
    for exp in roles:
        exp_text = _exp_to_text(exp)
        exp_emb = embedder.embed([exp_text])[0]
        semantic = _cosine(exp_emb, jd_emb)
        skill_score = _skill_overlap_score(exp.get("skills", []), jd_skills)

        # Tag bonus: if role type tag matches JD role type
        exp_tags = {t.lower() for t in exp.get("tags", [])}
        tag_bonus = 0.1 if jd_analysis.get("role_type", "") in exp_tags else 0.0

        score = semantic * 0.55 + skill_score * 0.35 + tag_bonus * 0.10
        scored.append((score, exp))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [exp for _, exp in scored[:top_n]]


def rank_projects(
    profile: dict,
    jd_text: str,
    jd_analysis: dict,
    embedder,
    top_n: int = 3,
) -> List[dict]:
    """Return top_n most relevant projects."""
    projects = profile.get("projects", [])
    if not projects:
        return []

    jd_emb = embedder.embed([jd_text[:2000]])[0]
    jd_skills = jd_analysis.get("required_skills", []) + jd_analysis.get("keywords", [])

    scored = []
    for proj in projects:
        proj_text = _proj_to_text(proj)
        proj_emb = embedder.embed([proj_text])[0]
        semantic = _cosine(proj_emb, jd_emb)
        skill_score = _skill_overlap_score(proj.get("skills", []), jd_skills)
        score = semantic * 0.60 + skill_score * 0.40
        scored.append((score, proj))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [proj for _, proj in scored[:top_n]]


def filter_skills(profile: dict, jd_analysis: dict) -> List[str]:
    """Return skills list prioritized by JD relevance."""
    technical = profile.get("skills", {}).get("technical", [])
    tools = profile.get("skills", {}).get("tools", [])
    all_skills = technical + [t for t in tools if t not in technical]

    jd_required = {s.lower() for s in jd_analysis.get("required_skills", [])}
    jd_keywords = {s.lower() for s in jd_analysis.get("keywords", [])}
    relevant = jd_required | jd_keywords

    matched = [s for s in all_skills if s.lower() in relevant]
    others = [s for s in all_skills if s.lower() not in relevant]
    return matched + others
