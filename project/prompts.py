from __future__ import annotations


def profile_summary_prompt(cv_text: str) -> str:
    return f"""
You are a strict CV analyzer.
Use ONLY the CV text below.
Return JSON with keys: summary, skills, key_experience.
- summary: max 80 words
- skills: list of explicit skills mentioned
- key_experience: list of 3 concise bullets
Do not invent facts.

CV TEXT:
{cv_text[:12000]}
""".strip()


def tailoring_prompt(job_text: str, cv_evidence_md: str) -> str:
    return f"""
You are a CV tailoring assistant.
Use ONLY the CV evidence and job description provided.
Never fabricate tools, experience, metrics, or achievements.
If a required skill is not supported in evidence, mark it as Missing.

Return markdown with sections exactly:
## Tailoring Plan
1) Recommended summary statement (1-2 lines)
2) Skills reorder and keyword guidance
3) Up to 3 rewritten bullets (STAR-style). Include metrics only if present in evidence.
4) Missing skills

Every factual claim must include citation like (p2).

JOB DESCRIPTION:
{job_text[:10000]}

CV EVIDENCE:
{cv_evidence_md[:12000]}
""".strip()
