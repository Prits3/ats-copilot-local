"""CV Builder + ATS Copilot — Clean 3-mode app."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from bullet_rewriter import generate_positioning_summary, holistic_cv_rewrite, rewrite_experience_bullets
from cv_advisor import ats_score, detailed_ats_suggestions, interpret_ats_score, missing_skills
from cv_generator import generate_cv_markdown, generate_pdf_bytes
from jd_analyzer import analyze_jd, extract_master_profile_from_cv, sanitize_profile
from job_fetcher import load_company_sources, scan_company_jobs
from llm_client import is_ollama_available, make_llm_fn
from matcher import Embedder, build_profile_store, rank_jobs, retrieve_cv_evidence
from profile import build_profile, extract_pdf_pages_from_bytes
from relevance_engine import filter_skills, rank_experiences, rank_projects

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CV Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Hide default streamlit header padding */
.block-container { padding-top: 2rem; }

/* Homepage cards */
.cv-card {
    border: 2px solid #e8e8e8;
    border-radius: 16px;
    padding: 36px 28px;
    text-align: center;
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: white;
    transition: border-color 0.2s, box-shadow 0.2s;
    margin-bottom: 8px;
}
.cv-card:hover {
    border-color: #ff4b4b;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.cv-card-icon { font-size: 2.4em; margin-bottom: 12px; }
.cv-card h3 { font-size: 1.25em; font-weight: 700; margin-bottom: 10px; color: #1a1a1a; }
.cv-card p { color: #666; font-size: 0.92em; line-height: 1.6; }

/* Diff view */
.bullet-original { background: #29200a; border: 1px solid #5a4000; border-radius: 6px; padding: 8px 12px; margin: 4px 0; font-size: 0.88em; color: #d4a854; }
.bullet-rewritten { background: #0d2b1a; border: 1px solid #1e5c30; border-radius: 6px; padding: 8px 12px; margin: 4px 0; font-size: 0.88em; color: #5cb87c; }
.bullet-label { font-size: 0.72em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; opacity: 0.65; display: block; margin-bottom: 4px; }

/* ATS score badge */
.ats-badge { font-size: 2em; font-weight: 800; }
.ats-good { color: #28a745; }
.ats-ok { color: #fd7e14; }
.ats-bad { color: #dc3545; }

/* Footer */
.footer { text-align: center; color: #999; font-size: 0.85em; padding: 48px 0 24px; }

/* Job analysis cards */
.match-card {
    background: #1e1e1e; border: 1px solid #333; border-radius: 10px;
    padding: 12px 16px; margin: 6px 0; font-size: 0.9em; line-height: 1.6;
}
.ats-row {
    display: flex; align-items: center; gap: 12px;
    background: #1a1a1a; border-radius: 8px; padding: 10px 14px; margin: 6px 0;
}
.ats-row-label { font-weight: 600; font-size: 0.88em; min-width: 130px; }
.ats-row-score { font-size: 0.85em; color: #999; min-width: 44px; text-align: right; }
.ats-row-fix { font-size: 0.85em; color: #ccc; flex: 1; }
.status-good { color: #28a745; }
.status-warn { color: #fd7e14; }
.status-bad  { color: #dc3545; }
.skill-chip {
    display: inline-block; background: #2a1a1a; border: 1px solid #dc354555;
    color: #ff8080; border-radius: 20px; padding: 2px 10px;
    font-size: 0.82em; margin: 3px 3px 3px 0;
}
.skill-chip-ok {
    display: inline-block; background: #0d2b1a; border: 1px solid #28a74555;
    color: #5cb85c; border-radius: 20px; padding: 2px 10px;
    font-size: 0.82em; margin: 3px 3px 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data
def cached_extract_pdf_pages(pdf_bytes: bytes) -> List[str]:
    return extract_pdf_pages_from_bytes(pdf_bytes)


@st.cache_data(ttl=30)
def cached_ollama_check(model: str) -> bool:
    return is_ollama_available(model)


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

def _empty_role() -> dict:
    return {"title": "", "company": "", "start_date": "", "end_date": "Present",
            "location": "", "bullets": "", "skills": ""}

def _empty_project() -> dict:
    return {"name": "", "description": "", "bullets": "", "skills": "", "url": ""}


def _init_state() -> None:
    defaults: dict = {
        "mode": None,           # None | "create" | "tailor" | "scan"
        # Shared
        "master_profile": None,
        "llm_provider": "openai" if os.environ.get("OPENAI_API_KEY") else "ollama",
        "llm_model": "gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "mistral:7b",
        # Create CV
        "create_personal": {},
        "create_roles": [_empty_role()],
        "create_projects": [_empty_project()],
        "create_education": [{"institution": "", "degree": "", "field": "",
                              "start_date": "", "end_date": "", "gpa": ""}],
        "create_skills_tech": "",
        "create_skills_tools": "",
        "create_skills_soft": "",
        "create_certs": "",
        "create_step": 1,
        # Tailor CV
        "tailor_jd": "",
        "tailor_cv_source": "profile",
        "tailor_cv_pdf_bytes": None,
        "tailor_positioning": "🔍 Auto-detect from JD",
        "tailor_result": None,     # {cv_md, ats, diff_pairs, jd_analysis}
        # Scan Jobs
        "scan_cv_text": "",
        "scan_profile": None,
        "scan_cv_index": None,
        "scan_prefs": None,
        "scan_jobs": [],
        "scan_ranked": [],
        "scan_stats": {},
        "scan_analysis": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

POSITIONING_MODES = {
    "🔍 Auto-detect from JD": "Automatically picks the best positioning based on the job description",
    "Analytical Problem Solver": "Rigorous analyst who turns data into clear decisions",
    "Startup Operator": "Scrappy executor who ships fast and drives growth",
    "AI-Focused Builder": "Technical builder who applies ML/AI to real problems",
    "Strategic Generalist": "Versatile thinker who connects strategy to execution",
    "Domain Expert": "Deep specialist with proven expertise in the field",
}

ROLE_TYPES = ["Data Analyst", "Business Analyst", "AI/ML Engineer", "Finance Analyst",
              "Product Analyst", "Software Engineer", "Product Manager", "Operations"]
INDUSTRIES = ["Fintech", "HealthTech", "E-Commerce", "SaaS/B2B", "AI/DeepTech",
              "Mobility", "PropTech", "Climate/GreenTech", "EdTech", "Other"]
SENIORITY_TYPES = ["Intern", "Junior", "Entry", "Mid"]
WORK_MODES = ["Remote", "Hybrid", "Onsite"]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: back button
# ──────────────────────────────────────────────────────────────────────────────

def _back_btn():
    if st.button("← Back to Home"):
        st.session_state["mode"] = None
        st.rerun()


def _llm_fn():
    return make_llm_fn(
        provider=st.session_state["llm_provider"],
        model=st.session_state["llm_model"],
    )


def _llm_available() -> bool:
    """Return True if the current LLM provider is reachable."""
    provider = st.session_state.get("llm_provider", "ollama")
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return cached_ollama_check(st.session_state.get("llm_model", "mistral:7b"))


def _suggest_positioning(jd_text: str) -> str:
    """Pick the best positioning mode based on JD keywords — no LLM needed."""
    low = jd_text.lower()
    if any(k in low for k in [
        "machine learning", "deep learning", "pytorch", "tensorflow", "llm",
        "large language model", "ml engineer", "data scientist", "neural network",
        "computer vision", "nlp", "reinforcement learning",
    ]):
        return "AI-Focused Builder"
    if any(k in low for k in [
        "product manager", "product roadmap", "go-to-market", "gtm", "b2b saas",
        "user research", "okr", "kpi alignment", "stakeholder management",
    ]):
        return "Strategic Generalist"
    if any(k in low for k in [
        "lead", "principal", "head of", "staff engineer", "10+ years", "8+ years",
        "domain expertise", "subject matter expert", "sme",
    ]):
        return "Domain Expert"
    if any(k in low for k in [
        "startup", "early-stage", "series a", "series b", "fast-paced",
        "wear many hats", "scrappy", "growth", "0 to 1", "scale fast",
    ]):
        return "Startup Operator"
    if any(k in low for k in [
        "data analyst", "business analyst", "sql", "dashboard", "reporting",
        "analytics", "insights", "kpi", "tableau", "power bi", "looker",
        "excel", "data-driven", "metrics",
    ]):
        return "Analytical Problem Solver"
    return "Analytical Problem Solver"


def _profile_from_cv_text(cv_text: str) -> dict:
    """Build a minimal master profile from raw CV text — no LLM needed."""
    from profile import extract_profile_signals
    signals = extract_profile_signals(cv_text)
    skills = signals.get("skills", [])

    # Extract bullet-like lines (lines starting with action indicators or long enough)
    import re
    lines = [l.strip() for l in cv_text.split("\n") if l.strip()]
    bullets = [
        l for l in lines
        if re.match(r"^[•\-–*]", l) or (len(l) > 30 and not l.isupper() and not l.endswith(":"))
    ][:30]

    # Try to detect name from first non-empty line
    name = lines[0] if lines else "Your Name"

    # Second line is often the tagline/subtitle if it doesn't look like contact info
    tagline = ""
    if len(lines) > 1:
        second = lines[1]
        if "@" not in second and not re.match(r"[\+\d]", second) and len(second) < 120:
            tagline = second

    # Extract email / phone with regex
    email = next((w for l in lines for w in l.split() if "@" in w), "")
    phone = next((re.search(r"[\+\d][\d\s\-\(\)]{7,}", l) for l in lines if re.search(r"[\+\d][\d\s\-\(\)]{7,}", l)), None)
    phone_str = phone.group() if phone else ""

    return {
        "personal": {"name": name, "tagline": tagline, "email": email, "phone": phone_str,
                     "linkedin": "", "github": "", "location": ""},
        "roles": [{
            "id": "role_1",
            "title": "Professional Experience",
            "company": "",
            "start_date": "",
            "end_date": "Present",
            "location": "",
            "bullets": bullets,
            "skills": skills,
            "tags": [],
        }],
        "projects": [],
        "education": [],
        "skills": {
            "technical": skills[:10],
            "tools": skills[10:] if len(skills) > 10 else [],
            "soft": [],
        },
        "certifications": [],
        "achievements": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# HOMEPAGE
# ══════════════════════════════════════════════════════════════════════════════

def _show_homepage():
    # LLM status (top right via sidebar)
    with st.sidebar:
        st.subheader("Settings")
        provider_options = ["openai", "ollama"]
        st.selectbox("LLM Provider", provider_options, key="llm_provider")
        st.text_input("Model", key="llm_model")
        provider = st.session_state.get("llm_provider", "ollama")
        model = st.session_state.get("llm_model", "")
        if provider == "openai":
            if os.environ.get("OPENAI_API_KEY"):
                st.success(f"OpenAI ready ({model or 'gpt-4o-mini'})")
            else:
                st.error("OPENAI_API_KEY not set — run: export OPENAI_API_KEY=sk-...")
        else:
            ok = cached_ollama_check(model or "mistral:7b")
            st.success("Ollama connected") if ok else st.warning("Ollama not running — switch to OpenAI")

    # Hero
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;font-size:2.6em;font-weight:800;'>Build a CV that actually gets you hired.</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#666;font-size:1.1em;max-width:600px;margin:0 auto 48px;'>"
        "Stop rewriting your resume for every job. Use AI to generate, tailor, and position your CV "
        "based on what companies actually care about."
        "</p>",
        unsafe_allow_html=True,
    )

    # 3 Cards
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="cv-card">
          <div>
            <div class="cv-card-icon">✏️</div>
            <h3>Create New CV</h3>
            <p>Build your master CV once. Add your experience, projects, and skills — this becomes your personal database for all future applications.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start from scratch", use_container_width=True, key="btn_create"):
            st.session_state["mode"] = "create"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="cv-card">
          <div>
            <div class="cv-card-icon">🎯</div>
            <h3>Tailor CV to a Job</h3>
            <p>Paste a job description and instantly generate a tailored CV — optimized for relevance, impact, and ATS. No manual editing needed.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Customize for a role", use_container_width=True, key="btn_tailor"):
            st.session_state["mode"] = "tailor"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="cv-card">
          <div>
            <div class="cv-card-icon">🔍</div>
            <h3>Find Startup Jobs (Berlin)</h3>
            <p>Share your CV and get matched with high-growth Berlin startups. Built for fast-moving teams looking for builders, not just applicants.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Match with startups", use_container_width=True, key="btn_scan"):
            st.session_state["mode"] = "scan"
            st.rerun()

    st.markdown(
        "<p class='footer'>Designed for people who don't just apply — they position.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CREATE CV
# ══════════════════════════════════════════════════════════════════════════════

def _show_create_cv():
    _back_btn()
    st.title("✏️ Create Your CV")
    st.caption("Fill in your details once. Your profile becomes the source for all tailored CVs.")

    step = st.session_state["create_step"]

    # Step progress
    steps = ["Personal Info", "Experience", "Projects", "Education & Skills", "Preview"]
    cols = st.columns(5)
    for i, (col, label) in enumerate(zip(cols, steps), 1):
        if i < step:
            col.markdown(f"✅ **{label}**")
        elif i == step:
            col.markdown(f"🔵 **{label}**")
        else:
            col.markdown(f"⬜ {label}")

    st.divider()

    # ── Step 1: Personal ──
    if step == 1:
        st.subheader("Personal Information")
        p = st.session_state.get("create_personal", {})
        c1, c2 = st.columns(2)
        p["name"] = c1.text_input("Full Name *", value=p.get("name", ""))
        p["email"] = c2.text_input("Email *", value=p.get("email", ""))
        p["phone"] = c1.text_input("Phone", value=p.get("phone", ""))
        p["location"] = c2.text_input("Location (e.g. Berlin, Germany)", value=p.get("location", ""))
        p["linkedin"] = c1.text_input("LinkedIn URL", value=p.get("linkedin", ""))
        p["github"] = c2.text_input("GitHub URL", value=p.get("github", ""))
        st.session_state["create_personal"] = p
        if st.button("Next →", type="primary"):
            if not p.get("name") or not p.get("email"):
                st.error("Name and email are required.")
            else:
                st.session_state["create_step"] = 2
                st.rerun()

    # ── Step 2: Experience ──
    elif step == 2:
        st.subheader("Work Experience")
        st.caption("Add your experiences — the most relevant ones will be selected for each job.")
        roles = st.session_state["create_roles"]

        for i, role in enumerate(roles):
            with st.expander(
                f"Experience {i+1}: {role.get('title', '') or 'Untitled'}"
                + (f" @ {role['company']}" if role.get("company") else ""),
                expanded=(i == len(roles) - 1),
            ):
                c1, c2 = st.columns(2)
                roles[i]["title"] = c1.text_input("Job Title", value=role.get("title", ""), key=f"rt_{i}")
                roles[i]["company"] = c2.text_input("Company", value=role.get("company", ""), key=f"rc_{i}")
                c3, c4, c5 = st.columns(3)
                roles[i]["start_date"] = c3.text_input("Start (e.g. Jun 2023)", value=role.get("start_date", ""), key=f"rs_{i}")
                roles[i]["end_date"] = c4.text_input("End (or Present)", value=role.get("end_date", "Present"), key=f"re_{i}")
                roles[i]["location"] = c5.text_input("Location", value=role.get("location", ""), key=f"rl_{i}")
                roles[i]["bullets"] = st.text_area(
                    "Bullets (one per line)",
                    value=role.get("bullets", ""),
                    height=120,
                    placeholder="Built dashboard tracking 12 KPIs, reducing reporting time by 4h/week\nWrote SQL queries joining 5 tables to identify $200K in annual waste",
                    key=f"rb_{i}",
                )
                roles[i]["skills"] = st.text_input(
                    "Skills used (comma-separated)",
                    value=role.get("skills", ""),
                    placeholder="Python, SQL, Power BI",
                    key=f"rsk_{i}",
                )
                if len(roles) > 1 and st.button("Remove", key=f"rm_{i}"):
                    st.session_state["create_roles"].pop(i)
                    st.rerun()

        st.session_state["create_roles"] = roles

        c_add, c_next = st.columns([1, 3])
        if c_add.button("+ Add Experience"):
            st.session_state["create_roles"].append(_empty_role())
            st.rerun()
        if c_next.button("Next →", type="primary"):
            st.session_state["create_step"] = 3
            st.rerun()

    # ── Step 3: Projects ──
    elif step == 3:
        st.subheader("Projects")
        st.caption("Add personal projects, academic work, or side projects.")
        projects = st.session_state["create_projects"]

        for i, proj in enumerate(projects):
            with st.expander(
                f"Project {i+1}: {proj.get('name', '') or 'Untitled'}",
                expanded=(i == len(projects) - 1),
            ):
                c1, c2 = st.columns(2)
                projects[i]["name"] = c1.text_input("Project Name", value=proj.get("name", ""), key=f"pn_{i}")
                projects[i]["url"] = c2.text_input("URL (optional)", value=proj.get("url", ""), key=f"pu_{i}")
                projects[i]["description"] = st.text_input(
                    "One-line description", value=proj.get("description", ""), key=f"pd_{i}"
                )
                projects[i]["bullets"] = st.text_area(
                    "Bullets (one per line)",
                    value=proj.get("bullets", ""),
                    height=100,
                    key=f"pb_{i}",
                )
                projects[i]["skills"] = st.text_input(
                    "Skills used",
                    value=proj.get("skills", ""),
                    placeholder="Python, Streamlit, NLP",
                    key=f"psk_{i}",
                )
                if len(projects) > 1 and st.button("Remove", key=f"prm_{i}"):
                    st.session_state["create_projects"].pop(i)
                    st.rerun()

        st.session_state["create_projects"] = projects

        c_add, _, c_back, c_next = st.columns([1, 2, 1, 1])
        if c_add.button("+ Add Project"):
            st.session_state["create_projects"].append(_empty_project())
            st.rerun()
        if c_back.button("← Back"):
            st.session_state["create_step"] = 2
            st.rerun()
        if c_next.button("Next →", type="primary"):
            st.session_state["create_step"] = 4
            st.rerun()

    # ── Step 4: Education & Skills ──
    elif step == 4:
        st.subheader("Education")
        edu_list = st.session_state["create_education"]
        for i, edu in enumerate(edu_list):
            with st.expander(f"Education {i+1}", expanded=True):
                c1, c2 = st.columns(2)
                edu_list[i]["institution"] = c1.text_input("Institution", value=edu.get("institution", ""), key=f"ei_{i}")
                edu_list[i]["field"] = c2.text_input("Field of Study", value=edu.get("field", ""), key=f"ef_{i}")
                c3, c4, c5 = st.columns(3)
                edu_list[i]["degree"] = c3.text_input("Degree (BSc/MSc)", value=edu.get("degree", ""), key=f"ed_{i}")
                edu_list[i]["start_date"] = c4.text_input("Start", value=edu.get("start_date", ""), key=f"es_{i}")
                edu_list[i]["end_date"] = c5.text_input("End / Present", value=edu.get("end_date", ""), key=f"ee_{i}")
                edu_list[i]["gpa"] = st.text_input("GPA (optional)", value=edu.get("gpa", ""), key=f"eg_{i}")
        st.session_state["create_education"] = edu_list

        st.divider()
        st.subheader("Skills")
        c1, c2, c3 = st.columns(3)
        st.session_state["create_skills_tech"] = c1.text_area(
            "Technical Skills",
            value=st.session_state.get("create_skills_tech", ""),
            placeholder="Python\nSQL\nMachine Learning",
            height=120,
        )
        st.session_state["create_skills_tools"] = c2.text_area(
            "Tools & Software",
            value=st.session_state.get("create_skills_tools", ""),
            placeholder="Power BI\nTableau\ndbt\nAirflow",
            height=120,
        )
        st.session_state["create_skills_soft"] = c3.text_area(
            "Soft Skills",
            value=st.session_state.get("create_skills_soft", ""),
            placeholder="Stakeholder Communication\nData Storytelling",
            height=120,
        )

        st.divider()
        st.subheader("Certifications & Achievements (optional)")
        st.session_state["create_certs"] = st.text_area(
            "One per line",
            value=st.session_state.get("create_certs", ""),
            height=80,
            placeholder="Google Data Analytics Certificate (2024)\nDean's List 2023",
        )

        c_back, c_next = st.columns([1, 4])
        if c_back.button("← Back"):
            st.session_state["create_step"] = 3
            st.rerun()
        if c_next.button("Build My CV →", type="primary"):
            st.session_state["master_profile"] = _build_profile_from_form()
            st.session_state["create_step"] = 5
            st.rerun()

    # ── Step 5: Preview ──
    elif step == 5:
        profile = st.session_state.get("master_profile")
        if not profile:
            st.error("Profile not built. Go back and fill in your details.")
            st.session_state["create_step"] = 1
            st.rerun()

        st.subheader("Your CV is ready")

        # Generate a generic CV (no JD tailoring)
        generic_jd = {"role_type": "default", "required_skills": [], "keywords": [],
                      "nice_to_have_skills": [], "tone": "analytical", "seniority": "junior",
                      "company_type": "startup", "industry": "technology"}
        all_skills = (
            profile.get("skills", {}).get("technical", []) +
            profile.get("skills", {}).get("tools", [])
        )
        cv_md = generate_cv_markdown(
            profile,
            profile.get("roles", [])[:4],
            profile.get("projects", [])[:3],
            all_skills,
            generic_jd,
            summary="",
        )

        col_preview, col_actions = st.columns([2, 1])
        with col_preview:
            with st.expander("CV Preview", expanded=True):
                st.markdown(cv_md)

        with col_actions:
            st.markdown("### Download")
            st.download_button(
                "Download Markdown",
                data=cv_md,
                file_name="my_cv.md",
                mime="text/markdown",
                use_container_width=True,
            )
            pdf_bytes = generate_pdf_bytes(
                profile, profile.get("roles", [])[:4],
                profile.get("projects", [])[:3], all_skills, generic_jd, summary=""
            )
            if pdf_bytes:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name="my_cv.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.divider()
            st.markdown("### Next Steps")
            if st.button("🎯 Tailor this CV to a job", use_container_width=True):
                st.session_state["mode"] = "tailor"
                st.rerun()
            if st.button("🔍 Find matching jobs", use_container_width=True):
                st.session_state["mode"] = "scan"
                st.rerun()
            if st.button("Edit profile", use_container_width=True):
                st.session_state["create_step"] = 1
                st.rerun()

        # Save profile reminder
        st.download_button(
            "Save Profile JSON (reuse later)",
            data=json.dumps(profile, indent=2),
            file_name="my_profile.json",
            mime="application/json",
        )


def _build_profile_from_form() -> dict:
    """Assemble master profile dict from create-CV form state."""
    def _parse_lines(text: str) -> list:
        return [l.strip() for l in text.strip().split("\n") if l.strip()]

    def _parse_csv(text: str) -> list:
        return [s.strip() for s in text.split(",") if s.strip()]

    roles = []
    for i, r in enumerate(st.session_state.get("create_roles", [])):
        if not r.get("title") and not r.get("company"):
            continue
        roles.append({
            "id": f"role_{i+1}",
            "title": r.get("title", ""),
            "company": r.get("company", ""),
            "start_date": r.get("start_date", ""),
            "end_date": r.get("end_date", "Present"),
            "location": r.get("location", ""),
            "bullets": _parse_lines(r.get("bullets", "")),
            "skills": _parse_csv(r.get("skills", "")),
            "tags": [],
        })

    projects = []
    for i, p in enumerate(st.session_state.get("create_projects", [])):
        if not p.get("name"):
            continue
        projects.append({
            "id": f"proj_{i+1}",
            "name": p.get("name", ""),
            "description": p.get("description", ""),
            "bullets": _parse_lines(p.get("bullets", "")),
            "skills": _parse_csv(p.get("skills", "")),
            "tags": [],
            "url": p.get("url", ""),
        })

    education = []
    for edu in st.session_state.get("create_education", []):
        if edu.get("institution"):
            education.append({**edu, "relevant_courses": []})

    tech = _parse_lines(st.session_state.get("create_skills_tech", ""))
    tools = _parse_lines(st.session_state.get("create_skills_tools", ""))
    soft = _parse_lines(st.session_state.get("create_skills_soft", ""))

    certs_raw = _parse_lines(st.session_state.get("create_certs", ""))
    certs = [c for c in certs_raw if not c.startswith("Dean") and "place" not in c.lower() and "hackathon" not in c.lower()]
    achievements = [c for c in certs_raw if c not in certs]

    return {
        "personal": st.session_state.get("create_personal", {}),
        "roles": roles,
        "projects": projects,
        "education": education,
        "skills": {"technical": tech, "tools": tools, "soft": soft},
        "certifications": certs,
        "achievements": achievements,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAILOR CV
# ══════════════════════════════════════════════════════════════════════════════

def _show_tailor_cv():
    _back_btn()
    st.title("🎯 Tailor CV to a Job")
    st.caption("Paste a job description — AI rewrites your CV for that specific role, optimized for ATS.")

    def _on_jd_change():
        pass  # auto-detect is handled at render time when "🔍 Auto-detect from JD" is selected

    col_input, col_output = st.columns([1, 1], gap="large")

    # ── Left: Inputs ──
    with col_input:
        st.subheader("Job Description")
        jd_text = st.text_area(
            "Paste the full job description",
            height=280,
            placeholder="Paste the job description here...",
            key="tailor_jd",
            on_change=_on_jd_change,
        )

        st.subheader("Your CV")
        cv_source_options = ["Use saved profile"]
        if st.session_state.get("master_profile"):
            cv_source_options = ["Use saved profile"] + ["Upload a different CV (PDF)"]
        else:
            cv_source_options = ["Upload CV (PDF)", "Load profile template"]

        cv_source = st.selectbox("CV source", cv_source_options)

        profile = st.session_state.get("master_profile")

        if cv_source == "Upload CV (PDF)" or cv_source == "Upload a different CV (PDF)":
            cv_file = st.file_uploader("Upload CV PDF", type=["pdf"], key="tailor_pdf")
            if cv_file:
                st.session_state["tailor_cv_pdf_bytes"] = cv_file.getvalue()
                profile = None  # will extract below
        elif cv_source == "Load profile template":
            if st.button("Load Template"):
                try:
                    with open("master_profile_template.json") as f:
                        st.session_state["master_profile"] = json.load(f)
                    profile = st.session_state["master_profile"]
                    st.success("Template loaded!")
                except FileNotFoundError:
                    st.error("Template file not found.")

        if profile:
            personal = profile.get("personal", {})
            st.success(
                f"Using: **{personal.get('name', 'Your profile')}** — "
                f"{len(profile.get('roles', []))} experiences, "
                f"{len(profile.get('projects', []))} projects"
            )

        st.subheader("Positioning")
        positioning = st.selectbox(
            "How should this CV position you?",
            list(POSITIONING_MODES.keys()),
            key="tailor_positioning",
        )
        if positioning == "🔍 Auto-detect from JD":
            if jd_text.strip():
                detected = _suggest_positioning(jd_text)
                st.caption(f"Will use: **{detected}** — {POSITIONING_MODES[detected]}")
            else:
                st.caption("Paste a job description above to auto-detect the best positioning.")
        else:
            st.caption(f"*{POSITIONING_MODES[positioning]}*")

        rewrite_on = st.toggle("Rewrite bullets with AI", value=True,
                               help="Rewrites your bullets to align with this specific role")
        summary_on = st.toggle("Generate summary paragraph", value=True)

        generate_btn = st.button("Generate Tailored CV", type="primary", use_container_width=True)

    # ── Right: Output ──
    with col_output:
        result = st.session_state.get("tailor_result")

        if generate_btn:
            if not jd_text.strip():
                st.error("Paste a job description first.")
            else:
                # Extract profile from PDF if needed
                if not profile and st.session_state.get("tailor_cv_pdf_bytes"):
                    with st.spinner("Extracting profile from PDF..."):
                        pages = cached_extract_pdf_pages(st.session_state["tailor_cv_pdf_bytes"])
                        cv_text = "\n".join(pages)
                        # Try LLM extraction first, fall back to regex-based extraction
                        extracted = extract_master_profile_from_cv(cv_text, _llm_fn())
                        if not extracted:
                            extracted = _profile_from_cv_text(cv_text)
                        extracted = sanitize_profile(extracted)
                        profile = extracted
                        st.session_state["master_profile"] = extracted

                if not profile:
                    st.error("No CV loaded. Upload a PDF or create your profile first.")
                    st.stop()

                # Always sanitize at generation time — even cached profiles may be stale
                profile = sanitize_profile(profile)

                embedder = get_embedder()
                fn = _llm_fn()

                with st.spinner("Analyzing job description..."):
                    jd_analysis = analyze_jd(jd_text, fn)

                progress = st.progress(0, text="Selecting best experiences...")
                selected_exps = rank_experiences(profile, jd_text, jd_analysis, embedder, top_n=4)
                selected_projs = rank_projects(profile, jd_text, jd_analysis, embedder, top_n=3)
                relevant_skills = filter_skills(profile, jd_analysis)

                effective_positioning = (
                    _suggest_positioning(jd_text)
                    if positioning == "🔍 Auto-detect from JD"
                    else positioning
                )

                # ── Holistic single-shot rewrite (preferred) ─────────────────
                diff_pairs = []
                summary = ""
                if rewrite_on and _llm_available():
                    # LLM is available — rewrite entire CV in one call
                    progress.progress(30, text="AI is rewriting your CV…")
                    rewrite_result = holistic_cv_rewrite(
                        profile, selected_exps, selected_projs,
                        jd_text, jd_analysis, effective_positioning, fn,
                    )
                    orig_bullets = {
                        e.get("title", ""): e.get("bullets", []) for e in selected_exps
                    }
                    selected_exps  = rewrite_result["experiences"]
                    selected_projs = rewrite_result["projects"]
                    summary        = rewrite_result.get("summary", "")

                    # Build diff pairs for "What Changed" tab
                    for exp in selected_exps:
                        title = exp.get("title", "")
                        orig = orig_bullets.get(title, [])
                        for i, new_b in enumerate(exp.get("bullets", [])):
                            old_b = orig[i] if i < len(orig) else new_b
                            diff_pairs.append((old_b, new_b, title))

                    # Generate summary with LLM if holistic didn't produce one
                    if not summary and summary_on:
                        progress.progress(70, text="Generating summary...")
                        summary = generate_positioning_summary(
                            profile, jd_analysis, effective_positioning, fn
                        )
                else:
                    # No LLM — use original bullets, try summary separately
                    diff_pairs = [(b, b, e.get("title", "")) for e in selected_exps for b in e.get("bullets", [])]
                    if summary_on:
                        summary = generate_positioning_summary(
                            profile, jd_analysis, effective_positioning, fn
                        )

                progress.progress(90, text="Assembling CV...")
                cv_md = generate_cv_markdown(
                    profile, selected_exps, selected_projs,
                    relevant_skills, jd_analysis, summary,
                )
                cv_ats = ats_score(cv_md, jd_text)
                pdf_bytes = generate_pdf_bytes(
                    profile, selected_exps, selected_projs,
                    relevant_skills, jd_analysis, summary,
                )
                progress.progress(100, text="Done!")
                time.sleep(0.3)
                progress.empty()

                st.session_state["tailor_result"] = {
                    "cv_md": cv_md,
                    "pdf_bytes": pdf_bytes,
                    "_debug_summary": summary,
                    "_debug_exp_count": len(selected_exps),
                    "_debug_proj_count": len(selected_projs),
                    "ats": cv_ats,
                    "diff_pairs": diff_pairs,
                    "jd_analysis": jd_analysis,
                    "profile": profile,
                    "selected_exps": selected_exps,
                    "selected_projs": selected_projs,
                    "relevant_skills": relevant_skills,
                    "summary": summary,
                }
                result = st.session_state["tailor_result"]

        if result:
            cv_ats = result["ats"]
            score_val = cv_ats.get("total", 0)
            css_class = "ats-good" if score_val >= 75 else ("ats-ok" if score_val >= 55 else "ats-bad")

            st.markdown(
                f"<div style='text-align:center;margin-bottom:16px;'>"
                f"<span class='ats-badge {css_class}'>{score_val}</span>"
                f"<span style='font-size:1em;color:#666;'> / 100 ATS Score</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Tabs: CV / What Changed / ATS Details
            tab_cv, tab_diff, tab_ats = st.tabs(["📄 CV", "🔄 What Changed", "📊 ATS Details"])

            with tab_cv:
                import base64 as _b64
                pdf = result.get("pdf_bytes")
                if pdf:
                    b64 = _b64.b64encode(pdf).decode()
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="750px" style="border:none;border-radius:8px;"></iframe>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(result["cv_md"])

            with tab_diff:
                jd_analysis = result.get("jd_analysis", {})
                role = jd_analysis.get('role_type', '').replace('_', ' ').title()
                seniority = jd_analysis.get('seniority', '').title()
                req_skills = ', '.join(jd_analysis.get('required_skills', [])[:6])
                if role:
                    st.caption(f"**{role}** · {seniority}" + (f" · {req_skills}" if req_skills else ""))
                st.divider()

                changed_pairs = [(o, r, t) for o, r, t in result.get("diff_pairs", []) if o != r]
                unchanged_pairs = [(o, r, t) for o, r, t in result.get("diff_pairs", []) if o == r and len(o) < 300]

                if changed_pairs:
                    seen_titles: set = set()
                    for orig, rewritten, title in changed_pairs:
                        if title and title not in seen_titles:
                            st.markdown(f"##### {title}")
                            seen_titles.add(title)
                        c1, c2 = st.columns(2)
                        c1.markdown(
                            f"<div class='bullet-original'>"
                            f"<span class='bullet-label'>Original</span>{orig}"
                            f"</div>", unsafe_allow_html=True
                        )
                        c2.markdown(
                            f"<div class='bullet-rewritten'>"
                            f"<span class='bullet-label'>Rewritten ✓</span>{rewritten}"
                            f"</div>", unsafe_allow_html=True
                        )
                else:
                    st.info("No bullets were rewritten — either rewrite was off or the CV had no structured bullets.")

                if unchanged_pairs:
                    with st.expander(f"Unchanged bullets ({len(unchanged_pairs)})"):
                        for orig, _, title in unchanged_pairs:
                            st.caption(f"**{title}** — {orig[:120]}{'…' if len(orig) > 120 else ''}")

                # Debug panel — shows what went into the CV
                with st.expander("🔍 Debug: what the AI generated", expanded=False):
                    st.caption(f"**Experiences used:** {result.get('_debug_exp_count', '?')}  |  "
                               f"**Projects used:** {result.get('_debug_proj_count', '?')}")
                    if result.get("_debug_summary"):
                        st.caption(f"**Generated summary:** {result['_debug_summary']}")
                    st.code(result["cv_md"], language="markdown")

                # Missing keywords
                miss = missing_skills(jd_text, result["cv_md"])
                if miss.get("missing"):
                    st.divider()
                    chips = " ".join(
                        f"<span style='background:#2a1a1a;border:1px solid #dc354555;color:#ff8080;"
                        f"border-radius:20px;padding:2px 10px;font-size:0.82em;margin:2px;display:inline-block;'>{s}</span>"
                        for s in miss["missing"]
                    )
                    st.markdown(f"**Missing keywords** — add these to your profile if you have them:", unsafe_allow_html=False)
                    st.markdown(chips, unsafe_allow_html=True)

            with tab_ats:
                ats_interp = interpret_ats_score(cv_ats)
                band_color = "#28a745" if ats_interp["band"] == "Strong" else ("#fd7e14" if ats_interp["band"] == "Good" else "#dc3545")
                st.markdown(
                    f"<p style='font-size:1.05em;'><span style='color:{band_color};font-weight:700;'>{ats_interp['band']}</span>"
                    f" — {ats_interp['meaning']}</p>",
                    unsafe_allow_html=True,
                )
                st.divider()
                suggestions = detailed_ats_suggestions(cv_ats, result["cv_md"])
                for s in suggestions:
                    status_color = {"good": "#28a745", "warn": "#fd7e14", "bad": "#dc3545"}[s["status"]]
                    pct = int(s["score"] / s["max"] * 100) if s["max"] else 0
                    st.markdown(
                        f"<div class='ats-row' style='margin-bottom:10px;'>"
                        f"<div style='min-width:130px;font-weight:600;font-size:0.9em;'>{s['category']}</div>"
                        f"<div style='flex:1;'>"
                        f"<div style='background:#333;border-radius:4px;height:6px;margin-bottom:4px;'>"
                        f"<div style='background:{status_color};width:{pct}%;height:6px;border-radius:4px;'></div></div>"
                        f"<span style='font-size:0.82em;color:#aaa;'>{s['fix']}</span>"
                        f"</div>"
                        f"<div style='min-width:52px;text-align:right;font-size:0.88em;color:{status_color};font-weight:700;'>"
                        f"{s['score']}/{s['max']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Downloads
            st.divider()
            pdf_data = result.get("pdf_bytes")
            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇ Download PDF",
                data=pdf_data or b"",
                file_name="tailored_cv.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=pdf_data is None,
            )
            dl2.download_button(
                "⬇ Download Markdown",
                data=result["cv_md"],
                file_name="tailored_cv.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("Fill in the job description and click **Generate Tailored CV**.")


# ══════════════════════════════════════════════════════════════════════════════
# SCAN JOBS
# ══════════════════════════════════════════════════════════════════════════════

def _show_scan_jobs():
    _back_btn()
    st.title("🔍 Find Startup Jobs in Berlin")
    st.caption("Scans 50+ Berlin startup career pages and ranks jobs against your CV.")

    from cv_advisor import build_why_match_and_action_plan, cv_improvement_suggestions, tailor_cv_with_ollama

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader("Your CV")

        source_options = ["Upload PDF"]
        if st.session_state.get("master_profile"):
            source_options = ["Use saved profile", "Upload PDF"]
        scan_source = st.radio("CV source", source_options)

        cv_ready = bool(st.session_state.get("scan_cv_index"))

        if scan_source == "Upload PDF":
            cv_file = st.file_uploader("Upload CV PDF", type=["pdf"])
            if cv_file:
                if st.button("Load CV", type="secondary"):
                    with st.spinner("Extracting text and building index..."):
                        try:
                            pages = cached_extract_pdf_pages(cv_file.getvalue())
                            cv_text = "\n".join(pages)
                            # Build profile without Ollama (just text + chunks)
                            from profile import chunk_text, extract_profile_signals
                            chunks = [{"text": t, "page": 0} for t in chunk_text(cv_text)]
                            skills = extract_profile_signals(cv_text).get("skills", [])
                            embedder = get_embedder()
                            _, store = build_profile_store(chunks, embedder=embedder)
                            st.session_state["scan_cv_text"] = cv_text
                            st.session_state["scan_profile"] = {"cv_text": cv_text, "chunks": chunks, "skills": skills}
                            st.session_state["scan_cv_index"] = {
                                "embedder": embedder, "store": store,
                                "profile_embedding": embedder.embed([cv_text[:2000]])[0],
                            }
                            cv_ready = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading CV: {e}")

        elif scan_source == "Use saved profile":
            profile = st.session_state["master_profile"]
            personal = profile.get("personal", {})
            if st.button("Use This Profile", type="secondary") or not cv_ready:
                with st.spinner("Building index from profile..."):
                    try:
                        # Build a readable text blob from the profile
                        lines = []
                        for r in profile.get("roles", []):
                            lines.append(f"{r.get('title','')} at {r.get('company','')}")
                            lines.extend(r.get("bullets", []))
                            lines.append("Skills: " + ", ".join(r.get("skills", [])))
                        for p in profile.get("projects", []):
                            lines.append(p.get("name", ""))
                            lines.extend(p.get("bullets", []))
                        skills_all = (
                            profile.get("skills", {}).get("technical", []) +
                            profile.get("skills", {}).get("tools", [])
                        )
                        lines.append("Skills: " + ", ".join(skills_all))
                        cv_text = "\n".join(lines)

                        chunks = [{"text": cv_text[i:i+900], "page": 0}
                                  for i in range(0, max(1, len(cv_text)), 780)]
                        embedder = get_embedder()
                        _, store = build_profile_store(chunks, embedder=embedder)
                        st.session_state["scan_cv_text"] = cv_text
                        st.session_state["scan_profile"] = {"cv_text": cv_text, "chunks": chunks, "skills": skills_all}
                        st.session_state["scan_cv_index"] = {
                            "embedder": embedder, "store": store,
                            "profile_embedding": embedder.embed([cv_text[:2000]])[0],
                        }
                        cv_ready = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Index error: {e}")

        if cv_ready:
            st.success("CV ready — you can now scan for jobs.")

        st.divider()
        st.subheader("Filters")
        role_types = st.multiselect("Role Types", ROLE_TYPES, default=["Data Analyst"])
        industries = st.multiselect("Industries", INDUSTRIES, default=[])
        seniority = st.multiselect("Seniority", SENIORITY_TYPES, default=["Junior", "Entry"])
        berlin_only = st.toggle("Berlin only", value=True)
        work_mode = st.multiselect("Work Mode", WORK_MODES, default=["Hybrid", "Remote"])
        keywords_text = st.text_input("Extra Keywords", placeholder="Python, SQL, Analytics")

        st.divider()
        companies = load_company_sources("company_sources.json")
        max_co = st.slider("Companies to scan", 1, len(companies), min(15, len(companies)))
        max_jobs = st.slider("Max jobs per company", 1, 15, 6)

        cv_indexed = bool(st.session_state.get("scan_cv_index"))
        if not cv_indexed:
            st.warning("Load your CV above to enable scanning.")
        scan_btn = st.button("Start Scan", type="primary", use_container_width=True,
                             disabled=not cv_indexed)

    with col_right:
        if scan_btn:
            if not st.session_state.get("scan_cv_index"):
                st.error("Load your CV first.")
            else:
                prefs = {
                    "role_types": role_types, "seniority": seniority,
                    "keywords": [k.strip() for k in keywords_text.split(",") if k.strip()],
                    "berlin_only": berlin_only, "work_mode": work_mode,
                    "industries": industries,
                }
                st.session_state["scan_prefs"] = prefs

                with st.spinner(f"Scanning {max_co} Berlin startup career pages..."):
                    try:
                        scanned = scan_company_jobs(
                            companies[:max_co],
                            max_jobs_per_company=max_jobs,
                            max_total_jobs=150, hard_timeout_s=12,
                        )

                        def _filter(job):
                            desc = (job.get("description_text", "") or "").lower()
                            loc = (job.get("location_guess", "") or "").lower()
                            if berlin_only and "berlin" not in loc and "berlin" not in desc:
                                return False
                            modes = [m.lower() for m in work_mode]
                            if modes and not any(m in desc or m in loc for m in modes):
                                return False
                            return True

                        filtered = [j for j in scanned if _filter(j)]
                        idx = st.session_state["scan_cv_index"]
                        p = st.session_state["scan_profile"]
                        ranking_kw = prefs["keywords"] + role_types + seniority
                        ranked = rank_jobs(
                            jobs=filtered,
                            profile_embedding=idx["profile_embedding"],
                            embedder=idx["embedder"],
                            profile_skills=p.get("skills", []),
                            interests_keywords=ranking_kw,
                        )
                        st.session_state["scan_jobs"] = filtered
                        st.session_state["scan_ranked"] = ranked[:10]
                        st.session_state["scan_stats"] = {
                            "scanned": len(scanned), "filtered": len(filtered)
                        }
                    except Exception as exc:
                        st.error(f"Scan failed: {exc}")

        ranked_jobs = st.session_state.get("scan_ranked", [])
        stats = st.session_state.get("scan_stats", {})

        if stats:
            c1, c2, c3 = st.columns(3)
            c1.metric("Companies Scanned", stats.get("scanned", 0) // max(1, max_jobs))
            c2.metric("Jobs Found", stats.get("filtered", 0))
            c3.metric("Top Matches", len(ranked_jobs))

        if ranked_jobs:
            st.subheader("Top Job Matches")
            rows = []
            for job in ranked_jobs:
                miss = missing_skills(job.get("description_text", ""), st.session_state.get("scan_cv_text", ""))
                job_ats = ats_score(st.session_state.get("scan_cv_text", ""), job.get("description_text", ""))
                rows.append({
                    "Match": f"{job.get('match_score', 0)}%",
                    "Title": job.get("title", ""),
                    "Company": job.get("company_guess", ""),
                    "Missing Skills": len(miss["missing"]),
                    "ATS": job_ats["total"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("Job Details")
            options = [
                f"{j.get('match_score', 0)}% — {j.get('title', '?')} @ {j.get('company_guess', '?')}"
                for j in ranked_jobs
            ]
            sel_idx = st.selectbox("Select a job to analyze", range(len(options)), format_func=lambda i: options[i])
            selected_job = ranked_jobs[sel_idx]

            st.markdown(
                f"**{selected_job.get('title', '')}** @ {selected_job.get('company_guess', '')}"
            )

            if st.button("Analyze this job", type="secondary", use_container_width=True):
                idx = st.session_state["scan_cv_index"]
                evidence = retrieve_cv_evidence(
                    store=idx["store"], embedder=idx["embedder"],
                    query=selected_job.get("description_text", ""), k=6,
                )
                miss = missing_skills(selected_job.get("description_text", ""), st.session_state.get("scan_cv_text", ""))
                job_ats = ats_score(st.session_state.get("scan_cv_text", ""), selected_job.get("description_text", ""))
                why_action = build_why_match_and_action_plan(evidence, miss["missing"])
                st.session_state["scan_analysis"] = {
                    "miss": miss, "ats": job_ats, "why_action": why_action,
                }

            analysis = st.session_state.get("scan_analysis")
            if analysis:
                score = analysis["ats"]["total"]
                css_class = "ats-good" if score >= 75 else ("ats-ok" if score >= 55 else "ats-bad")
                interp = interpret_ats_score(analysis["ats"])

                # Score header
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:8px;'>"
                    f"<span class='ats-badge {css_class}' style='font-size:2.4em;font-weight:800;'>{score}</span>"
                    f"<div><div style='font-size:0.9em;color:#999;'>/ 100 ATS Score</div>"
                    f"<div style='font-weight:700;font-size:1em;'>{interp['band']}</div>"
                    f"<div style='color:#aaa;font-size:0.85em;'>{interp['meaning']}</div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                t1, t2, t3 = st.tabs(["Why You Match", "What's Missing", "Fix My Score"])

                with t1:
                    why_lines = analysis["why_action"]["why_match"][:3]
                    if why_lines and why_lines[0] != "Limited CV evidence retrieved for this job match.":
                        st.caption("Strongest signals from your CV that match this role:")
                        for ln in why_lines:
                            clean = ln.split(") ", 1)[-1] if ") " in ln else ln
                            st.markdown(
                                f"<div class='match-card'>✅ {clean}</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("Load your CV in the panel above to see match evidence.")

                with t2:
                    miss = analysis["miss"]
                    matched = miss.get("matched", [])
                    missing = miss.get("missing", [])

                    if matched or missing:
                        st.caption(f"Required skills from JD — {len(matched)} matched, {len(missing)} missing")
                        chips = "".join(
                            f"<span class='skill-chip-ok'>✓ {s}</span>" for s in matched
                        ) + "".join(
                            f"<span class='skill-chip'>✗ {s}</span>" for s in missing
                        )
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.success("No obvious skill gaps detected.")

                    if missing:
                        st.caption(
                            f"**{len(missing)} missing:** If you have these skills, add them to your CV. "
                            "If not, the tailored CV generator can help reframe what you do have."
                        )

                with t3:
                    cv_txt = st.session_state.get("scan_cv_text", "")
                    suggestions = detailed_ats_suggestions(analysis["ats"], cv_txt)
                    for s in suggestions:
                        icon = "✅" if s["status"] == "good" else ("⚠️" if s["status"] == "warn" else "❌")
                        status_cls = f"status-{s['status']}"
                        pct = int(s["score"] / s["max"] * 100)
                        st.markdown(
                            f"<div class='ats-row'>"
                            f"<span class='ats-row-label'>{icon} {s['category']}</span>"
                            f"<span class='ats-row-score {status_cls}'>{s['score']}/{s['max']}</span>"
                            f"<span class='ats-row-fix'>{s['fix']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(score / 100, text=f"Overall: {score}/100")

                # Fix it for me
                st.divider()
                col_fix, col_apply = st.columns([2, 1])
                with col_fix:
                    if st.button(
                        "🎯 Fix it for me — Generate Tailored CV",
                        type="primary", use_container_width=True,
                    ):
                        st.session_state["tailor_jd"] = selected_job.get("description_text", "")
                        st.session_state["mode"] = "tailor"
                        st.rerun()
                    st.caption(
                        "Automatically rewrites your bullets, adds missing keywords (where truthful), "
                        "and generates an ATS-optimized PDF for this exact role."
                    )
                with col_apply:
                    if selected_job.get("url"):
                        st.link_button("Apply →", selected_job["url"], use_container_width=True)
        elif not scan_btn:
            st.info("Configure your filters on the left and click **Start Scan**.")


# ──────────────────────────────────────────────────────────────────────────────
# Routing — must come after all function definitions
# ──────────────────────────────────────────────────────────────────────────────

mode = st.session_state["mode"]

if mode is None:
    _show_homepage()
elif mode == "create":
    _show_create_cv()
elif mode == "tailor":
    _show_tailor_cv()
elif mode == "scan":
    _show_scan_jobs()
