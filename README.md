# Berlin Startup Job Hunter + CV Tailor + ATS Score

Streamlit app that:
- parses your CV PDF,
- scans Berlin company career pages from `company_sources.json`,
- ranks jobs against your profile,
- shows ATS score, missing skills, and action plan.

## Live App (after deploy)
Add your deployed Streamlit URL here:

`https://<your-app-name>.streamlit.app`

## Local Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Deploy (Option A)
This repo is deployable on Streamlit Community Cloud.

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud -> New app.
3. Select repo + branch `main`.
4. Main file path: `app.py`.
5. Deploy.

## Notes on LLM (Ollama)
- The app works without Ollama.
- In Step 4, `Use local LLM (Ollama)` is optional.
- On Streamlit Cloud, keep it OFF unless you host an accessible Ollama endpoint.

## Compliance
This tool is designed for user-managed company sources and public career pages.
It does not target restricted job boards (LinkedIn, Indeed, Glassdoor, StepStone).

## Project Files
- `app.py` - 4-step wizard UI
- `company_sources.json` - curated companies/career URLs
- `job_fetcher.py` - job fetching/parsing
- `profile.py` - CV extraction/redaction/profile build
- `matcher.py` - embeddings + ranking + retrieval
- `cv_advisor.py` - ATS score + missing skills + guidance
