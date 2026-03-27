"""FastAPI wrapper — exposes ATS Copilot logic as a REST API.

Start with:
    uvicorn api:app --reload --port 8000

Endpoints
---------
GET  /health                  — liveness check
POST /analyze-jd              — extract structured signal from a job description
POST /extract-profile         — parse raw CV text into a master profile JSON
POST /optimize                — full pipeline: select best exps + rewrite bullets
POST /ats-score               — ATS match score for a profile vs. JD
POST /generate-pdf            — generate PDF bytes from a profile
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

from bullet_rewriter import holistic_cv_rewrite
from cv_advisor import ats_score, detailed_ats_suggestions, missing_skills
from cv_generator import generate_pdf_bytes
from jd_analyzer import analyze_jd, extract_master_profile_from_cv, sanitize_profile
from llm_client import make_llm_fn
from relevance_engine import filter_skills, rank_experiences, rank_projects


def _profile_to_cv_text(profile: dict) -> str:
    """Flatten a structured profile dict to plain text for cv_advisor functions."""
    parts = []
    personal = profile.get("personal", {})
    for field in ("name", "email", "phone", "linkedin", "github", "location"):
        val = personal.get(field, "")
        if val:
            parts.append(val)
    for role in profile.get("roles", []):
        parts.append(f"{role.get('title', '')} {role.get('company', '')}")
        parts.extend(role.get("bullets", []))
        parts.extend(role.get("skills", []))
    for proj in profile.get("projects", []):
        parts.append(proj.get("name", ""))
        parts.append(proj.get("description", ""))
        parts.extend(proj.get("bullets", []))
        parts.extend(proj.get("skills", []))
    for edu in profile.get("education", []):
        parts.append(f"{edu.get('degree', '')} {edu.get('field', '')} {edu.get('institution', '')}")
    skills = profile.get("skills", {})
    for skill_list in skills.values():
        if isinstance(skill_list, list):
            parts.extend(skill_list)
    parts.extend(profile.get("achievements", []))
    parts.extend(profile.get("certifications", []))
    return "\n".join(p for p in parts if p)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ATS Copilot API",
    description="AI-powered CV optimisation for ATS targeting.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _get_llm_fn(provider: str = "gemini", model: Optional[str] = None):
    """Return an LLM callable. Raises 503 if the required API key is missing."""
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set.")
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set.")
    if provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not set.")
    return make_llm_fn(provider=provider, model=model)


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeJDRequest(BaseModel):
    jd_text: str
    provider: str = "openai"
    model: Optional[str] = None


class ExtractProfileRequest(BaseModel):
    cv_text: str
    provider: str = "openai"
    model: Optional[str] = None


class OptimizeRequest(BaseModel):
    """Full CV optimisation pipeline."""
    profile: Dict[str, Any]          # master profile JSON
    jd_text: str
    positioning: str = "Analytical Problem Solver"
    rewrite: bool = True             # run holistic LLM rewrite
    top_experiences: int = 4
    top_projects: int = 3
    provider: str = "openai"
    model: Optional[str] = None


class AtsScoreRequest(BaseModel):
    profile: Dict[str, Any]
    jd_text: str


class GeneratePdfRequest(BaseModel):
    profile: Dict[str, Any]
    selected_experiences: Optional[List[Dict[str, Any]]] = None
    selected_projects: Optional[List[Dict[str, Any]]] = None
    relevant_skills: Optional[Dict[str, Any]] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-jd")
def analyze_jd_endpoint(req: AnalyzeJDRequest):
    """Extract role_type, required_skills, keywords, etc. from a job description."""
    if not req.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text is required.")
    fn = _get_llm_fn(req.provider, req.model)
    result = analyze_jd(req.jd_text, fn)
    return result


@app.post("/extract-profile")
def extract_profile_endpoint(req: ExtractProfileRequest):
    """Parse raw CV text into a structured master profile JSON."""
    if not req.cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text is required.")
    fn = _get_llm_fn(req.provider, req.model)
    profile = extract_master_profile_from_cv(req.cv_text, fn)
    if not profile:
        raise HTTPException(
            status_code=422,
            detail="Could not extract a structured profile from the provided CV text.",
        )
    return {"profile": profile}


@app.post("/optimize")
def optimize_endpoint(req: OptimizeRequest):
    """Run the full ATS optimisation pipeline.

    Returns the selected experiences and projects with rewritten bullets, ATS
    score, missing keywords, and suggestions.
    """
    profile = sanitize_profile(req.profile)
    fn = _get_llm_fn(req.provider, req.model)

    jd_analysis = analyze_jd(req.jd_text, fn)

    # Rank and select best-fit experiences / projects
    try:
        from matcher import Embedder
        embedder = Embedder()
    except Exception:
        embedder = None

    selected_exps = rank_experiences(
        profile, req.jd_text, jd_analysis, embedder, top_n=req.top_experiences
    )
    selected_projs = rank_projects(
        profile, req.jd_text, jd_analysis, embedder, top_n=req.top_projects
    )
    relevant_skills = filter_skills(profile, jd_analysis)

    # Holistic LLM rewrite (optional)
    rewritten_exps = selected_exps
    summary = ""
    diff_pairs: list = []
    if req.rewrite:
        rewrite_result = holistic_cv_rewrite(
            profile, selected_exps, selected_projs,
            req.jd_text, jd_analysis, req.positioning, fn,
        )
        if rewrite_result:
            rewritten_exps = rewrite_result.get("experiences", selected_exps)
            summary = rewrite_result.get("summary", "")
            diff_pairs = rewrite_result.get("diff_pairs", [])

    # ATS score
    cv_text = _profile_to_cv_text(profile)
    score = ats_score(cv_text, req.jd_text)
    missing = missing_skills(req.jd_text, cv_text)
    suggestions = detailed_ats_suggestions(score, cv_text)

    return {
        "jd_analysis": jd_analysis,
        "selected_experiences": rewritten_exps,
        "selected_projects": selected_projs,
        "relevant_skills": relevant_skills,
        "ats_score": score,
        "missing_skills": missing,
        "suggestions": suggestions,
        "rewrite_summary": summary,
        "diff_pairs": diff_pairs,
    }


@app.post("/ats-score")
def ats_score_endpoint(req: AtsScoreRequest):
    """Return ATS match score and missing keywords for a profile vs. JD."""
    cv_text = _profile_to_cv_text(req.profile)
    score = ats_score(cv_text, req.jd_text)
    missing = missing_skills(req.jd_text, cv_text)
    suggestions = detailed_ats_suggestions(score, cv_text)
    return {
        "ats_score": score,
        "missing_skills": missing,
        "suggestions": suggestions,
    }


@app.post("/generate-pdf")
def generate_pdf_endpoint(req: GeneratePdfRequest):
    """Generate a PDF from a profile. Returns base64-encoded PDF bytes."""
    profile = sanitize_profile(req.profile)
    selected_exps = req.selected_experiences or profile.get("roles", [])[:4]
    selected_projs = req.selected_projects or profile.get("projects", [])[:3]
    relevant_skills = req.relevant_skills or profile.get("skills", {})

    try:
        pdf_bytes = generate_pdf_bytes(profile, selected_exps, selected_projs, relevant_skills)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    return {
        "pdf_base64": base64.b64encode(pdf_bytes).decode(),
        "size_bytes": len(pdf_bytes),
    }
