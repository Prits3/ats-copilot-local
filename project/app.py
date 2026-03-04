from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from job_import import import_jobs_from_manual_text, import_jobs_from_urls
from profile import build_profile_from_pages, extract_pdf_pages
from rag import EmbeddingModel, build_profile_vector_store, retrieve_relevant_cv_chunks
from scoring import compute_ats_score, compute_job_fit_score, compute_missing_skills
from tailoring import generate_tailoring_report
from utils import safe_filename

st.set_page_config(page_title="Berlin Startup Job Hunter + CV Tailor", layout="wide")


def _init_state() -> None:
    defaults = {
        "profile": None,
        "embedder": None,
        "cv_store": None,
        "jobs": [],
        "ranked_jobs": [],
        "tailoring": {},
        "debug": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _read_uploaded_pdf(uploaded_file) -> List[str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        return extract_pdf_pages(path)
    finally:
        os.unlink(path)


def _profile_ready() -> bool:
    return bool(st.session_state.get("profile") and st.session_state.get("embedder") and st.session_state.get("cv_store"))


def _rank_jobs() -> List[Dict]:
    profile = st.session_state["profile"]
    embedder: EmbeddingModel = st.session_state["embedder"]
    jobs = st.session_state["jobs"]

    if not jobs:
        return []

    profile_emb = embedder.embed([profile["cv_text"]])[0]
    rows = []
    for idx, job in enumerate(jobs):
        desc = job.get("description_text", "")
        if not desc:
            continue
        job_emb = embedder.embed([desc])[0]
        fit = compute_job_fit_score(
            profile_embedding=profile_emb,
            job_embedding=job_emb,
            profile_skills=profile.get("skills", []),
            job_text=desc,
        )
        missing = compute_missing_skills(desc, profile["cv_text"])
        row = {
            "job_id": idx,
            "title": job.get("title", "Unknown"),
            "company": job.get("company_guess", "Unknown"),
            "location": job.get("location_guess", "Unknown"),
            "source": job.get("source", ""),
            "url": job.get("url", ""),
            "JobFitScore": fit["total"],
            "EmbedSim": fit["embedding_similarity"],
            "MissingSkillsCount": len(missing["missing"]),
            "Error": job.get("error", ""),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["JobFitScore"], reverse=True)
    st.session_state["ranked_jobs"] = rows
    return rows


_init_state()

st.title("Berlin Startup Job Hunter + CV Tailor + ATS Score")
st.caption("Fully local workflow: sentence-transformers embeddings + Ollama for tailoring.")
st.sidebar.checkbox("Debug mode", key="debug")

if st.sidebar.button("Clear Session"):
    for key in ["profile", "embedder", "cv_store", "jobs", "ranked_jobs", "tailoring"]:
        st.session_state[key] = None if key in {"profile", "embedder", "cv_store"} else [] if key in {"jobs", "ranked_jobs"} else {}
    st.rerun()

tab1, tab2, tab3, tab4 = st.tabs([
    "1) Upload Profile",
    "2) Import Jobs",
    "3) Ranked Jobs + Filters",
    "4) Tailor CV + ATS",
])

with tab1:
    st.subheader("Upload Master CV")
    cv_file = st.file_uploader("Master CV PDF", type=["pdf"], accept_multiple_files=False)
    model_name = st.text_input("Ollama model", value="mistral:7b")

    if st.button("Process Profile"):
        if not cv_file:
            st.error("Please upload a CV PDF.")
        else:
            with st.spinner("Extracting CV, building profile summary, embedding chunks..."):
                try:
                    pages = _read_uploaded_pdf(cv_file)
                    profile = build_profile_from_pages(pages, model=model_name)
                    embedder, store = build_profile_vector_store(profile["chunks"])
                    st.session_state["profile"] = profile
                    st.session_state["embedder"] = embedder
                    st.session_state["cv_store"] = store
                    st.success(f"Profile loaded. {len(profile['pages'])} pages, {len(profile['chunks'])} chunks. Backend: {store.kind}")
                except Exception as exc:  # noqa: BLE001
                    if st.session_state["debug"]:
                        st.exception(exc)
                    else:
                        st.error(f"Could not process profile: {exc}")

    profile = st.session_state.get("profile")
    if profile:
        st.markdown("#### Profile Summary")
        st.write(profile["profile_summary"].get("summary", ""))
        st.write("Skills:", ", ".join(profile.get("skills", [])) or "None detected")
        st.write("Key experience:")
        for item in profile["profile_summary"].get("key_experience", [])[:3]:
            st.markdown(f"- {item}")

with tab2:
    st.subheader("Import Jobs")
    st.info("Compliance: Do not use restricted job boards. Paste company career pages or public job pages.")

    url_input = st.text_area("Paste job URLs (one per line)")
    manual_jobs = st.text_area(
        "Paste job descriptions directly (use line '===' between multiple jobs)",
        height=200,
    )

    if st.button("Import Jobs"):
        jobs: List[Dict] = []
        with st.spinner("Importing jobs..."):
            try:
                jobs.extend(import_jobs_from_urls(url_input, sleep_s=1.0))
                jobs.extend(import_jobs_from_manual_text(manual_jobs, source_label="manual"))
                st.session_state["jobs"] = jobs
                st.success(f"Imported {len(jobs)} jobs.")
            except Exception as exc:  # noqa: BLE001
                if st.session_state["debug"]:
                    st.exception(exc)
                else:
                    st.error(f"Import failed: {exc}")

    jobs = st.session_state.get("jobs", [])
    if jobs:
        preview = pd.DataFrame(
            [
                {
                    "title": j.get("title", ""),
                    "company": j.get("company_guess", ""),
                    "location": j.get("location_guess", ""),
                    "source": j.get("source", ""),
                    "url": j.get("url", ""),
                    "error": j.get("error", ""),
                }
                for j in jobs
            ]
        )
        st.dataframe(preview, use_container_width=True)

with tab3:
    st.subheader("Ranked Jobs")
    if not _profile_ready():
        st.warning("Upload and process profile first.")
    elif not st.session_state.get("jobs"):
        st.warning("Import jobs first.")
    else:
        if st.button("Compute Ranking"):
            with st.spinner("Computing embeddings and ranking jobs..."):
                try:
                    _rank_jobs()
                except Exception as exc:  # noqa: BLE001
                    if st.session_state["debug"]:
                        st.exception(exc)
                    else:
                        st.error(f"Ranking failed: {exc}")

        rows = st.session_state.get("ranked_jobs", [])
        if rows:
            c1, c2, c3 = st.columns(3)
            location_kw = c1.text_input("Filter location contains")
            only_remote = c2.checkbox("Remote only")
            junior_only = c3.checkbox("Junior only")

            filtered = rows
            if location_kw:
                filtered = [r for r in filtered if location_kw.lower() in (r.get("location", "").lower())]
            if only_remote:
                filtered = [r for r in filtered if "remote" in (r.get("location", "").lower())]
            if junior_only:
                filtered = [
                    r
                    for r in filtered
                    if ("junior" in r.get("title", "").lower() or "entry" in r.get("title", "").lower())
                ]

            df = pd.DataFrame(filtered)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download Ranked Jobs CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="ranked_jobs.csv",
                mime="text/csv",
            )

with tab4:
    st.subheader("Tailor CV for Selected Job + ATS")
    if not _profile_ready():
        st.warning("Upload and process profile first.")
    elif not st.session_state.get("ranked_jobs"):
        st.warning("Compute ranking first.")
    else:
        ranked = st.session_state["ranked_jobs"]
        options = [f"{r['job_id']} | {r['title']} | {r['company']}" for r in ranked]
        selected = st.selectbox("Select job", options=options)
        selected_id = int(selected.split("|")[0].strip())
        selected_job = st.session_state["jobs"][selected_id]

        k = st.slider("Top-k CV chunks for evidence", min_value=3, max_value=12, value=6)
        model_name = st.text_input("Ollama model for tailoring", value="mistral:7b", key="tailor_model")

        if st.button("Generate Tailoring Report"):
            with st.spinner("Retrieving CV evidence, computing ATS, generating tailoring plan..."):
                try:
                    profile = st.session_state["profile"]
                    embedder = st.session_state["embedder"]
                    store = st.session_state["cv_store"]
                    job_text = selected_job.get("description_text", "")

                    retrieved = retrieve_relevant_cv_chunks(store, embedder, job_text, k=k)
                    missing = compute_missing_skills(job_text, profile["cv_text"])
                    ats = compute_ats_score(profile["cv_text"], job_text)
                    tailoring = generate_tailoring_report(
                        job_text=job_text,
                        retrieved_chunks=retrieved,
                        missing_skills=missing["missing"],
                        model=model_name,
                    )

                    st.session_state["tailoring"][selected_id] = {
                        "retrieved": retrieved,
                        "missing": missing,
                        "ats": ats,
                        "tailoring": tailoring,
                        "job": selected_job,
                    }
                except Exception as exc:  # noqa: BLE001
                    if st.session_state["debug"]:
                        st.exception(exc)
                    else:
                        st.error(f"Tailoring failed: {exc}")

        data = st.session_state["tailoring"].get(selected_id)
        if data:
            st.markdown("### ATS Score")
            ats = data["ats"]
            st.write(f"**ATS Score: {ats['total']} / 100**")
            st.json(
                {
                    "contact_score": ats["contact_score"],
                    "section_score": ats["section_score"],
                    "parsability_score": ats["parsability_score"],
                    "keyword_score": ats["keyword_score"],
                    "impact_score": ats["impact_score"],
                    "contact_checks": ats["contact_checks"],
                    "heading_hits": ats["heading_hits"],
                }
            )

            st.markdown("### Missing Skills")
            st.write(", ".join(data["missing"]["missing"]) if data["missing"]["missing"] else "No obvious missing skills from configured list.")

            st.markdown("### Tailoring Report")
            t = data["tailoring"]
            if t.get("source") == "heuristic" and t.get("warning"):
                st.warning(f"LLM fallback used: {t['warning']}")
            st.markdown(t["report"])

            st.markdown("### CV Evidence")
            for ch in data["retrieved"]:
                st.markdown(f"- (p{ch.get('page', '?')}) score={ch.get('score', 0):.4f}")
                st.code(ch.get("text", "")[:300], language="text")

            out_name = safe_filename(f"tailoring_{data['job'].get('title', 'job')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            st.download_button(
                "Download Tailoring Report (.txt)",
                data["tailoring"]["report"].encode("utf-8"),
                file_name=out_name,
                mime="text/plain",
            )

st.markdown("---")
st.caption("Compliance: Uses user-provided URLs/text. Do not scrape restricted job boards (LinkedIn, Indeed, Glassdoor, StepStone).")
