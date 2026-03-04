from __future__ import annotations

import time
from typing import Dict, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from utils import guess_company, guess_location, normalize_whitespace

RESTRICTED_DOMAINS = {"linkedin.com", "indeed.com", "glassdoor.com", "stepstone.de"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


def _is_restricted(url: str) -> bool:
    host = urlparse(url).netloc.lower().replace("www.", "")
    return any(host == d or host.endswith(f".{d}") for d in RESTRICTED_DOMAINS)


def _extract_main_text(soup: BeautifulSoup) -> str:
    selectors = [
        "main",
        "article",
        "[class*='job']",
        "[id*='job']",
        "[class*='description']",
        "[id*='description']",
        "[class*='content']",
        "[id*='content']",
    ]
    chunks: List[str] = []
    for sel in selectors:
        for node in soup.select(sel):
            txt = normalize_whitespace(node.get_text(" ", strip=True))
            if len(txt) >= 120:
                chunks.append(txt)
    if chunks:
        return max(chunks, key=len)

    # Fallback: body without common boilerplate tags
    for bad in soup.select("nav, footer, script, style, noscript, header"):
        bad.extract()
    body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
    return normalize_whitespace(body)


def _extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    return "Unknown Job Title"


def fetch_job_url(url: str, timeout: int = 15) -> Dict:
    if _is_restricted(url):
        return {
            "source": "url",
            "url": url,
            "title": "Restricted domain",
            "company_guess": "Unknown",
            "location_guess": "Unknown",
            "description_text": "",
            "error": "Domain is restricted by compliance policy.",
        }

    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = _extract_title(soup)
        description = _extract_main_text(soup)
        if len(description) < 120:
            return {
                "source": "url",
                "url": url,
                "title": title,
                "company_guess": guess_company("", url=url),
                "location_guess": "Unknown",
                "description_text": description,
                "error": "Extracted text is very short.",
            }
        return {
            "source": "url",
            "url": url,
            "title": title,
            "company_guess": guess_company(description, url=url),
            "location_guess": guess_location(description),
            "description_text": description,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "url",
            "url": url,
            "title": "Fetch failed",
            "company_guess": "Unknown",
            "location_guess": "Unknown",
            "description_text": "",
            "error": str(exc),
        }


def import_jobs_from_urls(url_lines: str, sleep_s: float = 1.0) -> List[Dict]:
    urls = [u.strip() for u in (url_lines or "").splitlines() if u.strip()]
    out: List[Dict] = []
    for i, url in enumerate(urls):
        out.append(fetch_job_url(url))
        if i < len(urls) - 1:
            time.sleep(sleep_s)
    return out


def import_jobs_from_manual_text(manual_text: str, source_label: str = "manual") -> List[Dict]:
    text = (manual_text or "").strip()
    if not text:
        return []

    # Split by marker line for multiple jobs.
    blocks = [b.strip() for b in text.split("\n===\n") if b.strip()]
    out: List[Dict] = []
    for idx, block in enumerate(blocks, start=1):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        title = lines[0][:120] if lines else f"Manual Job {idx}"
        desc = normalize_whitespace(block)
        out.append(
            {
                "source": source_label,
                "url": "",
                "title": title,
                "company_guess": guess_company(block),
                "location_guess": guess_location(block),
                "description_text": desc,
                "error": "",
            }
        )
    return out
