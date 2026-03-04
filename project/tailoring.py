from __future__ import annotations

from typing import Dict, List

from prompts import tailoring_prompt
from utils import call_ollama


def format_cv_evidence(retrieved_chunks: List[Dict]) -> str:
    if not retrieved_chunks:
        return "- No CV evidence retrieved."
    lines = []
    for ch in retrieved_chunks:
        txt = (ch.get("text", "") or "").replace("\n", " ").strip()
        page = ch.get("page", "?")
        score = ch.get("score", 0.0)
        lines.append(f"- (p{page}, score={score:.4f}) {txt[:350]}")
    return "\n".join(lines)


def heuristic_tailoring(job_text: str, retrieved_chunks: List[Dict], missing_skills: List[str]) -> str:
    if not retrieved_chunks:
        return (
            "## Tailoring Plan\n"
            "1) Recommended summary statement (1-2 lines)\n"
            "- Candidate has relevant baseline profile; align wording directly with the target role.\n\n"
            "2) Skills reorder and keyword guidance\n"
            "- Prioritize skills explicitly required by the JD only if present in CV evidence.\n\n"
            "3) Up to 3 rewritten bullets (STAR-style)\n"
            "- No CV evidence retrieved to safely rewrite bullets with citations.\n\n"
            "4) Missing skills\n"
            f"- {', '.join(missing_skills) if missing_skills else 'No obvious missing skills from configured list.'}"
        )

    bullets = []
    for ch in retrieved_chunks[:3]:
        txt = ch.get("text", "").strip().replace("\n", " ")
        p = ch.get("page", "?")
        bullets.append(f"- Action/Context from CV evidence: {txt[:180]}... (p{p})")

    return (
        "## Tailoring Plan\n"
        "1) Recommended summary statement (1-2 lines)\n"
        "- Data-focused professional aligned to role requirements, emphasizing proven tools and outcomes from prior projects (see evidence citations).\n\n"
        "2) Skills reorder and keyword guidance\n"
        "- Move JD-aligned proven skills to top of Skills section; keep missing skills in a separate learning section.\n\n"
        "3) Up to 3 rewritten bullets (STAR-style)\n"
        + "\n".join(bullets)
        + "\n\n4) Missing skills\n"
        + f"- {', '.join(missing_skills) if missing_skills else 'No obvious missing skills from configured list.'}"
    )


def generate_tailoring_report(
    job_text: str,
    retrieved_chunks: List[Dict],
    missing_skills: List[str],
    model: str = "mistral:7b",
) -> Dict:
    evidence_md = format_cv_evidence(retrieved_chunks)
    prompt = tailoring_prompt(job_text=job_text, cv_evidence_md=evidence_md)
    result = call_ollama(prompt=prompt, model=model, timeout=180)

    if result.get("ok") and result.get("text", "").strip():
        return {
            "source": "ollama",
            "report": result["text"],
            "evidence_md": evidence_md,
        }

    fallback = heuristic_tailoring(job_text, retrieved_chunks, missing_skills)
    return {
        "source": "heuristic",
        "report": fallback,
        "evidence_md": evidence_md,
        "warning": result.get("error", "Ollama unavailable"),
    }
