# ATS Copilot

Live app: https://ats-copilot-local-24eeonythnfyexllks9sms.streamlit.app

AI-powered CV builder and ATS optimiser. Upload your CV, paste a job description, and get a tailored, rewritten CV with an ATS match score.

## What it does

- **Build CV** — create a professional CV from scratch with a guided form
- **Tailor CV** — upload your existing CV PDF, paste a JD, and get rewritten bullets optimised for that role
- **ATS Score** — keyword match score, missing skills, and improvement suggestions
- **PDF Export** — download a clean, formatted CV as a PDF

## LLM Providers

The app supports multiple LLM backends. Groq is recommended (free, fast).

| Provider | Default model | Get key |
|----------|--------------|---------|
| Groq | llama-3.3-70b-versatile | console.groq.com |
| Gemini | gemini-2.0-flash-lite | aistudio.google.com |
| OpenAI | gpt-4o-mini | platform.openai.com |
| Ollama | mistral:7b | local only |

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=
```

Run the app:

```bash
python3 -m streamlit run app.py
```

## API (FastAPI)

A REST API is also available:

```bash
uvicorn api:app --reload --port 8000
```

Endpoints:
- `GET /health`
- `POST /analyze-jd`
- `POST /extract-profile`
- `POST /optimize`
- `POST /ats-score`
- `POST /generate-pdf`

## Deploy to Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch `main`, main file `app.py`
4. Add your `GROQ_API_KEY` under App secrets
5. Deploy

## Project Files

- `app.py` — Streamlit UI
- `api.py` — FastAPI wrapper
- `bullet_rewriter.py` — LLM bullet rewriting
- `cv_generator.py` — PDF generation
- `cv_advisor.py` — ATS scoring and suggestions
- `jd_analyzer.py` — job description parsing
- `relevance_engine.py` — experience/project ranking
- `matcher.py` — semantic embeddings
- `llm_client.py` — unified LLM client (Groq, Gemini, OpenAI, Ollama)
