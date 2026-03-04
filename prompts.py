SYSTEM = """You are an ATS assistant. Be strict, evidence-based, and concise.
Never mention protected attributes. Only use the provided CV evidence + job description."""

def build_generation_prompt(jd: str, candidate_name: str, evidence_chunks_md: str, score_breakdown: dict) -> str:
    return f"""
{SYSTEM}

JOB DESCRIPTION:
{jd}

CANDIDATE: {candidate_name}
SCORE BREAKDOWN: {score_breakdown}

EVIDENCE (with page citations):
{evidence_chunks_md}

Tasks:
1) Write a 5-bullet "Why this candidate fits" (each bullet must cite a page like (p2)).
2) Write a 3-bullet "Risks / gaps" (each bullet must cite a page OR say "Not found in CV evidence").
3) Generate 6 interview questions targeted to gaps + role requirements.
Return in clean markdown.
"""
