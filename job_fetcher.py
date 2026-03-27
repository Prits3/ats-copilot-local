from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Dict, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BLOCKED_DOMAINS = {"linkedin.com", "indeed.com", "glassdoor.com", "stepstone.de"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


def load_company_sources(path: str = "company_sources.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    companies = data.get("companies", [])
    return [c for c in companies if c.get("careers_url")]


def is_blocked_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().replace("www.", "")
    return any(host == d or host.endswith(f".{d}") for d in BLOCKED_DOMAINS)


def _fetch(url: str, timeout: int = 5) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


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
    texts = []
    for sel in selectors:
        for node in soup.select(sel):
            txt = " ".join(node.get_text(" ", strip=True).split())
            if len(txt) > 120:
                texts.append(txt)
    if texts:
        return max(texts, key=len)

    for bad in soup.select("nav, footer, script, style, noscript, header"):
        bad.extract()
    body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
    return " ".join(body.split())


def _extract_title(soup: BeautifulSoup, fallback: str = "Unknown Role") -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    return fallback


def _extract_job_links_from_careers_page(base_url: str, html: str, limit: int = 12) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = " ".join(a.get_text(" ", strip=True).split())
        if not href:
            continue
        full = urljoin(base_url, href)
        low = (href + " " + text).lower()
        if not any(k in low for k in ["job", "career", "position", "opening", "vacanc", "apply"]):
            continue
        if full in seen:
            continue
        seen.add(full)
        links.append({"url": full, "title_hint": text[:140] or "Role"})
        if len(links) >= limit:
            break
    return links


def _scan_greenhouse_json(company_url: str) -> List[Dict]:
    # Supports URLs like: https://boards.greenhouse.io/company
    if "greenhouse.io" not in company_url:
        return []
    parsed = urlparse(company_url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return []
    token = parts[-1]
    api = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
    try:
        data = _fetch(api, timeout=15).json()
        jobs = []
        for j in data.get("jobs", []):
            jobs.append(
                {
                    "title": j.get("title", "Unknown Role"),
                    "url": j.get("absolute_url", company_url),
                    "location": (j.get("location") or {}).get("name", "Unknown"),
                    "description_text": "",
                    "source": "greenhouse_api",
                }
            )
        return jobs
    except Exception:
        return []


def _scan_lever_json(company_url: str) -> List[Dict]:
    # Supports URLs like: https://jobs.lever.co/company
    if "lever.co" not in company_url:
        return []
    parsed = urlparse(company_url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return []
    token = parts[-1]
    api = f"https://api.lever.co/v0/postings/{token}?mode=json"
    try:
        data = _fetch(api, timeout=15).json()
        jobs = []
        for j in data:
            jobs.append(
                {
                    "title": j.get("text", "Unknown Role"),
                    "url": j.get("hostedUrl", company_url),
                    "location": (j.get("categories") or {}).get("location", "Unknown"),
                    "description_text": " ".join(
                        [
                            (j.get("descriptionPlain") or ""),
                            (j.get("lists", [{}])[0].get("content", "") if j.get("lists") else ""),
                        ]
                    ).strip(),
                    "source": "lever_api",
                }
            )
        return jobs
    except Exception:
        return []


def fetch_job_detail(job_url: str, title_hint: str = "Unknown Role", timeout: int = 8) -> Dict:
    if is_blocked_url(job_url):
        return {
            "url": job_url,
            "title": title_hint,
            "location_guess": "Unknown",
            "description_text": "",
            "error": "Blocked domain by compliance policy.",
        }
    try:
        r = _fetch(job_url, timeout=timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        title = _extract_title(soup, fallback=title_hint)
        desc = _extract_main_text(soup)
        return {
            "url": job_url,
            "title": title,
            "location_guess": "Berlin" if "berlin" in desc.lower() else "Unknown",
            "description_text": desc,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "url": job_url,
            "title": title_hint,
            "location_guess": "Unknown",
            "description_text": "",
            "error": str(exc),
        }


def _scan_one_company(company: Dict, max_jobs: int, deadline: float) -> List[Dict]:
    """Fetch jobs for a single company. Returns list of job dicts."""
    if time.time() > deadline:
        return []
    name = company.get("name", "Unknown")
    location = company.get("location", "Berlin")
    careers_url = company.get("careers_url", "")
    if not careers_url or not careers_url.startswith("http") or is_blocked_url(careers_url):
        return []

    # Fast path: Greenhouse / Lever JSON APIs
    ats_jobs = _scan_greenhouse_json(careers_url) + _scan_lever_json(careers_url)
    if ats_jobs:
        out = []
        for aj in ats_jobs[:max_jobs]:
            out.append({
                "source": aj.get("source", "ats_api"),
                "url": aj.get("url", careers_url),
                "title": aj.get("title", "Unknown Role"),
                "company_guess": name,
                "location_guess": aj.get("location", location) or location,
                "description_text": aj.get("description_text", ""),
                "error": "",
            })
        return out

    # Slow path: scrape careers page then each job link
    try:
        remaining = max(1, int(deadline - time.time()))
        r = _fetch(careers_url, timeout=min(5, remaining))
        links = _extract_job_links_from_careers_page(careers_url, r.text, limit=max_jobs)
    except Exception:
        return []

    out = []
    with ThreadPoolExecutor(max_workers=min(len(links), 6)) as pool:
        futs = {
            pool.submit(fetch_job_detail, lk["url"], lk["title_hint"], 5): lk
            for lk in links
            if time.time() < deadline
        }
        for fut in as_completed(futs, timeout=max(1, deadline - time.time())):
            try:
                detail = fut.result()
                out.append({
                    "source": "career_page",
                    "url": detail["url"],
                    "title": detail["title"],
                    "company_guess": name,
                    "location_guess": detail["location_guess"] or location,
                    "description_text": detail["description_text"],
                    "error": detail["error"],
                })
            except Exception:
                pass
    return out


def scan_company_jobs(
    companies: List[Dict],
    delay_s: float = 0.0,
    max_jobs_per_company: int = 6,
    max_total_jobs: int = 120,
    hard_timeout_s: int = 12,
) -> List[Dict]:
    deadline = time.time() + hard_timeout_s
    jobs: List[Dict] = []

    valid = [
        c for c in companies
        if c.get("careers_url", "").startswith("http") and not is_blocked_url(c.get("careers_url", ""))
    ]

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {
            pool.submit(_scan_one_company, company, max_jobs_per_company, deadline): company
            for company in valid
        }
        for fut in as_completed(futs, timeout=max(1, hard_timeout_s)):
            if time.time() > deadline or len(jobs) >= max_total_jobs:
                break
            try:
                results = fut.result()
                jobs.extend(results)
            except Exception:
                pass

    return jobs[:max_total_jobs]
