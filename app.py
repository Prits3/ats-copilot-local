from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st

from cv_advisor import (
    ats_score,
    build_why_match_and_action_plan,
    cv_improvement_suggestions,
    interpret_ats_score,
    missing_skills,
    tailor_cv_with_ollama,
)
from job_fetcher import load_company_sources, scan_company_jobs
from matcher import Embedder, build_profile_store, rank_jobs, retrieve_cv_evidence
from profile import build_profile, extract_pdf_pages_from_bytes

ROLE_TYPES = [
    "Data Analyst",
    "Business Analyst",
    "AI Engineer",
    "Finance Analyst",
    "Product Analyst",
]
SENIORITY_TYPES = ["Intern", "Junior", "Entry", "Mid"]
WORK_MODES = ["Remote", "Hybrid", "Onsite"]


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data
def cached_extract_pdf_pages(pdf_bytes: bytes) -> List[str]:
    return extract_pdf_pages_from_bytes(pdf_bytes)


def _init_state() -> None:
    defaults = {
        "current_step": 1,
        "cv_text": "",
        "profile": None,
        "cv_index": None,  # {"embedder": ..., "store": ..., "profile_embedding": ...}
        "prefs": None,
        "jobs": [],
        "ranked_jobs": [],
        "scan_stats": {},
        "analysis_by_job_id": {},
        "debug": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _goto(step: int) -> None:
    st.session_state["current_step"] = step
    st.rerun()


def _step1_done() -> bool:
    return bool(st.session_state.get("cv_text") and st.session_state.get("cv_index"))


def _step2_done() -> bool:
    return bool(st.session_state.get("prefs"))


def _step3_done() -> bool:
    return bool(st.session_state.get("scan_stats"))


def _interest_filter(job: Dict, prefs: Dict) -> bool:
    # Keep filters permissive so scanning doesn't end up with zero jobs too often.
    desc = (job.get("description_text", "") or "").lower()
    location = (job.get("location_guess", "") or "").lower()

    if prefs.get("berlin_only", True):
        if "berlin" not in location and "berlin" not in desc:
            return False

    work_mode = [m.lower() for m in prefs.get("work_mode", [])]
    if work_mode and not any(m in desc or m in location for m in work_mode):
        return False

    return True


def _status_line() -> None:
    s1 = "✅" if _step1_done() else "⬜"
    s2 = "✅" if _step2_done() else "⬜"
    s3 = "✅" if _step3_done() else "⬜"
    s4 = "✅" if st.session_state.get("ranked_jobs") else "⬜"
    st.markdown(f"{s1} Step 1  →  {s2} Step 2  →  {s3} Step 3  →  {s4} Step 4")


_init_state()
st.set_page_config(page_title="Berlin Startup Job Hunter + CV Tailor + ATS Score", layout="wide")
st.title("Berlin Startup Job Hunter + CV Tailor + ATS Score")
st.caption("Smooth 4-step wizard: Upload CV -> Preferences -> Scan -> Rank + Action Plan")
_status_line()

st.sidebar.checkbox("Debug mode", key="debug")
if st.sidebar.button("Reset Wizard"):
    for key in ["current_step", "cv_text", "profile", "cv_index", "prefs", "jobs", "ranked_jobs", "scan_stats", "analysis_by_job_id"]:
        if key == "current_step":
            st.session_state[key] = 1
        elif key == "cv_text":
            st.session_state[key] = ""
        elif key in {"profile", "cv_index", "prefs"}:
            st.session_state[key] = None
        elif key in {"jobs", "ranked_jobs"}:
            st.session_state[key] = []
        else:
            st.session_state[key] = {}
    st.rerun()

nav_cols = st.columns(4)
if nav_cols[0].button("Go Step 1"):
    _goto(1)
if nav_cols[1].button("Go Step 2", disabled=not _step1_done()):
    _goto(2)
if nav_cols[2].button("Go Step 3", disabled=not _step2_done()):
    _goto(3)
if nav_cols[3].button("Go Step 4", disabled=not _step3_done()):
    _goto(4)

step = st.session_state["current_step"]

if step == 1:
    st.subheader("Step 1 — Upload CV")
    cv_file = st.file_uploader("Upload one CV PDF", type=["pdf"], accept_multiple_files=False)
    if st.button("Process CV", type="primary"):
        if not cv_file:
            st.error("Upload a CV first.")
        else:
            with st.spinner("Parsing CV and building index..."):
                try:
                    pdf_bytes = cv_file.getvalue()
                    pages = cached_extract_pdf_pages(pdf_bytes)
                    profile = build_profile(pages, model="mistral:7b")
                    embedder = get_embedder()
                    _, store = build_profile_store(profile["chunks"], embedder=embedder)
                    profile_embedding = embedder.embed([profile["cv_text"]])[0]
                    st.session_state["profile"] = profile
                    st.session_state["cv_text"] = profile["cv_text"]
                    st.session_state["cv_index"] = {
                        "embedder": embedder,
                        "store": store,
                        "profile_embedding": profile_embedding,
                    }
                    st.success("Step 1 completed.")
                    _goto(2)
                except Exception as exc:  # noqa: BLE001
                    if st.session_state["debug"]:
                        st.exception(exc)
                    else:
                        st.error(f"Could not process CV: {exc}")

elif step == 2:
    if not _step1_done():
        st.info("Complete Step 1 first.")
    else:
        st.subheader("Step 2 — Set Preferences")
        role_types = st.multiselect("Role types", ROLE_TYPES, default=["Data Analyst"])
        seniority = st.multiselect("Seniority", SENIORITY_TYPES, default=["Junior", "Entry"])
        keywords_text = st.text_input("Keywords (comma-separated)", value="Python, SQL, Power BI, Tableau")
        berlin_only = st.toggle("Berlin only", value=True)
        work_mode = st.multiselect("Work mode", WORK_MODES, default=["Hybrid", "Remote"])

        if st.button("Save Preferences", type="primary"):
            st.session_state["prefs"] = {
                "role_types": role_types,
                "seniority": seniority,
                "keywords": [k.strip() for k in keywords_text.split(",") if k.strip()],
                "berlin_only": berlin_only,
                "work_mode": work_mode,
            }
            st.success("Step 2 completed.")
            _goto(3)

elif step == 3:
    if not _step2_done():
        st.info("Complete Step 2 first.")
    else:
        st.subheader("Step 3 — Scan Companies & Collect Jobs")
        companies = load_company_sources("company_sources.json")
        st.write(f"Loaded companies: **{len(companies)}**")
        max_companies = st.slider("Max companies to scan", 1, max(1, len(companies)), min(10, max(1, len(companies))))
        max_jobs_per_company = st.slider("Max jobs per company", 1, 15, 6)
        hard_timeout_s = st.slider("Scan timeout (seconds)", 20, 180, 75)

        if st.button("Start Scan", type="primary"):
            with st.spinner("Scanning career pages..."):
                try:
                    scanned = scan_company_jobs(
                        companies[:max_companies],
                        delay_s=0.2,
                        max_jobs_per_company=max_jobs_per_company,
                        max_total_jobs=120,
                        hard_timeout_s=hard_timeout_s,
                    )
                    filtered = [j for j in scanned if _interest_filter(j, st.session_state["prefs"])]
                    st.session_state["jobs"] = filtered
                    st.session_state["scan_stats"] = {
                        "scanned_total": len(scanned),
                        "filtered_total": len(filtered),
                    }

                    # Auto-rank so Step 4 is ready immediately.
                    idx = st.session_state["cv_index"]
                    p = st.session_state["profile"]
                    prefs = st.session_state["prefs"]
                    ranking_keywords = prefs.get("keywords", []) + prefs.get("role_types", []) + prefs.get("seniority", [])
                    ranked = rank_jobs(
                        jobs=filtered,
                        profile_embedding=idx["profile_embedding"],
                        embedder=idx["embedder"],
                        profile_skills=p.get("skills", []),
                        interests_keywords=ranking_keywords,
                    )
                    st.session_state["ranked_jobs"] = ranked[:10]
                    st.success(
                        f"Step 3 completed. scanned={len(scanned)}, filtered={len(filtered)}, ranked={len(st.session_state['ranked_jobs'])}"
                    )
                    _goto(4)
                except Exception as exc:  # noqa: BLE001
                    if st.session_state["debug"]:
                        st.exception(exc)
                    else:
                        st.error(f"Scanning failed: {exc}")

        if st.session_state.get("scan_stats"):
            st.info(
                f"Last scan: scanned={st.session_state['scan_stats'].get('scanned_total', 0)}, "
                f"filtered={st.session_state['scan_stats'].get('filtered_total', 0)}"
            )

elif step == 4:
    if not _step3_done():
        st.info("Complete Step 3 first.")
    else:
        st.subheader("Step 4 — Rank Jobs + Action Plan")
        profile = st.session_state["profile"]
        ranked_jobs = st.session_state.get("ranked_jobs", [])

        st.markdown("### Your Profile")
        headline = " • ".join((profile.get("skills", [])[:4] or ["Profile extracted"]))
        st.write(headline)
        st.write("Skills:", ", ".join(profile.get("skills", [])) or "No skills detected")

        if not st.session_state.get("jobs"):
            st.warning("Scan finished but no jobs matched your filters. Go back to Step 2 and loosen preferences.")
        elif not ranked_jobs:
            st.warning("No ranked jobs available yet. Go back to Step 3 and scan again.")
        else:
            st.markdown("### Top Job Matches")
            rows = []
            for job in ranked_jobs:
                miss = missing_skills(job.get("description_text", ""), st.session_state["cv_text"])
                job_ats = ats_score(st.session_state["cv_text"], job.get("description_text", ""))
                rows.append(
                    {
                        "Job ID": job.get("job_id"),
                        "Job Title": job.get("title"),
                        "Company": job.get("company_guess"),
                        "Match Score": job.get("match_score"),
                        "Missing Skills Count": len(miss["missing"]),
                        "ATS Score": job_ats["total"],
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            st.markdown("### Job Details")
            options = [f"{j['job_id']} | {j.get('title', 'Unknown')} | {j.get('company_guess', 'Unknown')}" for j in ranked_jobs]
            selected = st.selectbox("Select a job", options=options)
            job_id = int(selected.split("|")[0].strip())
            selected_job = next((j for j in ranked_jobs if j.get("job_id") == job_id), None)

            if selected_job:
                use_llm = st.toggle("Use local LLM (Ollama)", value=False)
                k = st.slider("Top-k CV evidence", 3, 12, 6)
                idx = st.session_state["cv_index"]

                # Always compute baseline (non-LLM) analysis so Step 4 consistently shows suggestions.
                if job_id not in st.session_state["analysis_by_job_id"]:
                    evidence = retrieve_cv_evidence(
                        store=idx["store"],
                        embedder=idx["embedder"],
                        query=selected_job.get("description_text", ""),
                        k=k,
                    )
                    miss = missing_skills(selected_job.get("description_text", ""), st.session_state["cv_text"])
                    job_ats = ats_score(st.session_state["cv_text"], selected_job.get("description_text", ""))
                    why_action = build_why_match_and_action_plan(evidence, miss["missing"])
                    st.session_state["analysis_by_job_id"][job_id] = {
                        "job": selected_job,
                        "evidence": evidence,
                        "missing": miss,
                        "ats": job_ats,
                        "why_action": why_action,
                        "llm": {"ok": False, "text": ""},
                        "use_llm": False,
                    }

                if st.button("Generate Action Plan", type="primary"):
                    evidence = retrieve_cv_evidence(
                        store=idx["store"],
                        embedder=idx["embedder"],
                        query=selected_job.get("description_text", ""),
                        k=k,
                    )
                    miss = missing_skills(selected_job.get("description_text", ""), st.session_state["cv_text"])
                    job_ats = ats_score(st.session_state["cv_text"], selected_job.get("description_text", ""))
                    why_action = build_why_match_and_action_plan(evidence, miss["missing"])

                    llm_output = {"ok": False, "text": ""}
                    if use_llm:
                        llm_output = tailor_cv_with_ollama(selected_job.get("description_text", ""), evidence, model="mistral:7b")

                    st.session_state["analysis_by_job_id"][job_id] = {
                        "job": selected_job,
                        "evidence": evidence,
                        "missing": miss,
                        "ats": job_ats,
                        "why_action": why_action,
                        "llm": llm_output,
                        "use_llm": use_llm,
                    }

                analysis = st.session_state["analysis_by_job_id"].get(job_id)
                if analysis:
                    st.markdown("#### Why you match")
                    for ln in analysis["why_action"]["why_match"][:3]:
                        st.markdown(f"- {ln}")

                    st.markdown("#### Missing skills")
                    st.write(", ".join(analysis["missing"]["missing"]) or "No obvious missing skills.")

                    st.markdown("#### Action plan")
                    for ln in analysis["why_action"]["action_plan"]:
                        st.markdown(f"- {ln}")

                    st.markdown("#### CV improvement checklist")
                    for tip in cv_improvement_suggestions(ranked_jobs[:3], st.session_state["cv_text"]):
                        st.markdown(f"- {tip}")

                    st.markdown("#### ATS score")
                    st.write(f"**ATS Score: {analysis['ats']['total']} / 100**")
                    ats_interpretation = interpret_ats_score(analysis["ats"])
                    st.write(f"**Meaning:** {ats_interpretation['band']} — {ats_interpretation['meaning']}")
                    st.markdown("**How to improve this ATS score**")
                    for tip in ats_interpretation["tips"]:
                        st.markdown(f"- {tip}")
                    st.json(analysis["ats"])

                    if analysis["use_llm"]:
                        if analysis["llm"].get("ok"):
                            st.markdown("#### LLM suggestions")
                            st.markdown(analysis["llm"]["text"])
                        else:
                            st.info("Couldn’t connect to Ollama. Run: `ollama serve`")
                    else:
                        st.info("LLM suggestions disabled (turn on if Ollama is running locally).")

                    if selected_job.get("url"):
                        st.link_button("Apply", selected_job["url"])
                else:
                    st.warning("No analysis generated yet for this job.")
