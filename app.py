import os
import tempfile
from typing import Dict, List

import pandas as pd
import streamlit as st

from rag import LocalRAG, extract_pdf_pages
from scoring import compute_ats_score, score_candidate

IMPACT_WORDS = ["improved", "increased", "reduced", "optimized", "saved", "grew", "%", "roi", "revenue", "cost"]


def uploaded_pdf_to_pages(uploaded_file) -> List[str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
    try:
        pages = extract_pdf_pages(pdf_path)
    finally:
        os.unlink(pdf_path)
    return pages


def build_local_report(candidate_name: str, jd_text: str, score: Dict, retrieved: List[Dict]) -> str:
    if not retrieved:
        return (
            f"### Candidate: {candidate_name}\n\n"
            "#### Why this candidate fits\n"
            "- No matching evidence found in retrieved CV chunks.\n\n"
            "#### Risks / gaps\n"
            "- Insufficient CV evidence retrieved for this JD.\n\n"
            "#### Interview questions\n"
            "1. Walk me through your most relevant project for this role.\n"
            "2. Which tools from this JD have you used recently, and where?\n"
            "3. Describe a project where you delivered measurable impact.\n"
            "4. How do you approach debugging and root-cause analysis?\n"
            "5. How do you prioritize tasks when deadlines conflict?\n"
            "6. What part of this role do you expect will be hardest first 90 days?"
        )

    fits = []
    for r in retrieved:
        txt = r["text"].replace("\n", " ").strip()
        if not txt:
            continue
        fits.append(f"- (p{r['page']}) {txt[:180]}...")
        if len(fits) >= 5:
            break

    jd_skills = score.get("jd_skills", [])
    cv_skills = set(score.get("cv_skills", []))
    missing = [s for s in jd_skills if s not in cv_skills]
    risks = []
    for m in missing[:3]:
        risks.append(f"- Missing explicit evidence for `{m}` in top retrieved chunks.")
    if not risks:
        risks.append("- Strong keyword alignment in retrieved evidence; validate depth during interview.")

    # Add one risk from signal quality if possible
    evidence_text = " ".join(r["text"].lower() for r in retrieved)
    if not any(w in evidence_text for w in IMPACT_WORDS):
        risks.append("- Limited quantified impact words detected in retrieved evidence.")

    question_skills = missing[:]
    if len(question_skills) < 6:
        for s in jd_skills:
            if s not in question_skills:
                question_skills.append(s)
            if len(question_skills) >= 6:
                break
    while len(question_skills) < 6:
        question_skills.append("role requirements")

    questions = [
        f"{i + 1}. Tell me about your hands-on experience with {question_skills[i]}."
        for i in range(6)
    ]

    return (
        f"### Candidate: {candidate_name}\n\n"
        "#### Why this candidate fits\n"
        + "\n".join(fits)
        + "\n\n#### Risks / gaps\n"
        + "\n".join(risks[:3])
        + "\n\n#### Interview questions\n"
        + "\n".join(questions)
    )


st.set_page_config(page_title="ATS Copilot (Local RAG)", layout="wide")
st.title("ATS Copilot — Fully Local CV Screening (RAG)")

jd_text = st.text_area("Paste Job Description", height=220)
uploaded = st.file_uploader("Upload CV PDFs", type=["pdf"], accept_multiple_files=True)
k = st.slider("Top-k evidence chunks", 3, 10, 6)

if st.button("Rank Candidates"):
    if not jd_text.strip():
        st.error("Please paste a Job Description.")
        st.stop()
    if not uploaded:
        st.error("Please upload at least one CV PDF.")
        st.stop()

    rag = LocalRAG()
    rows = []
    details = {}

    for f in uploaded:
        pages = uploaded_pdf_to_pages(f)
        cv_text = "\n".join(pages)
        index, metas = rag.build_index(pages)
        retrieved = rag.retrieve(index, metas, jd_text, k=k)
        score = score_candidate(jd_text, retrieved)
        ats = compute_ats_score(cv_text, jd_text)

        candidate_name = f.name
        rows.append(
            {
                "Candidate": candidate_name,
                "Score": score["total"],
                "ATS Score": ats["total"],
                "Overlap": score["overlap"],
                "Skills Found": ", ".join(score["cv_skills"][:12]),
            }
        )

        details[candidate_name] = {
            "score": score,
            "ats": ats,
            "retrieved": retrieved,
            "report": build_local_report(candidate_name, jd_text, score, retrieved),
        }

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    st.subheader("Ranking")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="candidate_ranking.csv",
        mime="text/csv",
    )

    st.subheader("Candidate Details")
    choice = st.selectbox("Select candidate", df["Candidate"].tolist())
    st.markdown(details[choice]["report"])

    st.markdown("#### ATS Breakdown")
    ats = details[choice]["ats"]
    st.write(f"ATS Score: **{ats['total']} / 100**")
    st.write(
        "Contact/Basics:",
        f"{ats['contact_score']}/15",
        "| Sections:",
        f"{ats['section_score']}/15",
        "| Keyword Match:",
        f"{ats['keyword_score']}/30",
        "| Readability:",
        f"{ats['readability_score']}/20",
        "| Quantified Impact:",
        f"{ats['impact_score']}/20",
    )
    if ats["matched_keywords"]:
        st.write("Matched JD keywords:", ", ".join(ats["matched_keywords"]))

    st.markdown("#### Retrieved evidence")
    retrieved = details[choice]["retrieved"]
    if not retrieved:
        st.write("No evidence retrieved for this candidate.")
    else:
        for r in retrieved:
            st.markdown(f"- (p{r['page']}) score={r['score']:.4f}")
            st.code(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""), language="text")
