# Berlin Startup Job Hunter + CV Tailor + ATS Score (Local)

A fully local Streamlit app to:
- import startup job posts from user-provided URLs or pasted descriptions,
- rank jobs against a master CV,
- generate a safe tailoring plan (no fabricated experience),
- compute ATS score and missing skills.

## Features
- Local embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Local LLM: Ollama (`mistral:7b` default)
- Vector store backend:
  - prefer FAISS when available
  - automatically fall back to local ChromaDB if FAISS is unavailable
- CSV export for ranked jobs
- TXT export for per-job tailoring report

## Project Structure
- `app.py` - Streamlit UI and orchestration
- `profile.py` - CV import, chunking, summary extraction
- `job_import.py` - URL/manual job ingestion and parsing
- `rag.py` - embedding + vector store abstraction (FAISS/Chroma)
- `scoring.py` - JobFit + ATS + missing skills scoring
- `tailoring.py` - local tailoring report generation
- `prompts.py` - LLM prompts
- `utils.py` - shared helpers
- `sample_jobs.json` - example job payloads

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama setup
```bash
ollama pull mistral:7b
```

### Run
```bash
streamlit run app.py
```

Open: `http://localhost:8501`

## How to Use
1. Tab **Upload Profile**:
   - upload master CV PDF
   - click **Process Profile**
2. Tab **Import Jobs**:
   - paste job URLs (one per line) and/or manual job descriptions
   - for multiple manual jobs, separate with `===`
3. Tab **Ranked Jobs + Filters**:
   - click **Compute Ranking**
   - use filters (location/remote/junior)
   - export CSV
4. Tab **Tailor CV + ATS**:
   - select a ranked job
   - click **Generate Tailoring Report**
   - view ATS breakdown, missing skills, citations, and download report

## Compliance
This tool uses **user-provided job descriptions/URLs** and does **not** scrape restricted job boards (LinkedIn, Indeed, Glassdoor, StepStone).
Please respect each website's Terms of Service and robots policies.

## Notes
- If Ollama is unavailable, the app falls back to deterministic heuristic summaries/tailoring.
- Tailoring output is constrained to CV evidence and includes page citations where possible.
- The app avoids inventing achievements or metrics.

## Sample Jobs
`sample_jobs.json` contains example records you can paste/import manually.
