from __future__ import annotations

import json
import time
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


def _fetch(url: str, timeout: int = 8) -> requests.Response:
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


def scan_company_jobs(
    companies: List[Dict],
    delay_s: float = 0.2,
    max_jobs_per_company: int = 8,
    max_total_jobs: int = 120,
    hard_timeout_s: int = 90,
) -> List[Dict]:
    jobs: List[Dict] = []
    started = time.time()
    for i, company in enumerate(companies):
        if (time.time() - started) > hard_timeout_s:
            break
        if len(jobs) >= max_total_jobs:
            break
        name = company.get("name", "Unknown")
        location = company.get("location", "Berlin")
        careers_url = company.get("careers_url", "")
        if not careers_url:
            continue
        if not careers_url.startswith("http"):
            continue
        if is_blocked_url(careers_url):
            continue

        ats_jobs = _scan_greenhouse_json(careers_url) + _scan_lever_json(careers_url)
        if ats_jobs:
            for aj in ats_jobs[:max_jobs_per_company]:
                if len(jobs) >= max_total_jobs:
                    break
                jobs.append(
                    {
                        "source": aj.get("source", "ats_api"),
                        "url": aj.get("url", careers_url),
                        "title": aj.get("title", "Unknown Role"),
                        "company_guess": name,
                        "location_guess": aj.get("location", location) or location,
                        "description_text": aj.get("description_text", ""),
                        "error": "",
                    }
                )
            if i < len(companies) - 1:
                time.sleep(delay_s)
            continue

        try:
            r = _fetch(careers_url, timeout=8)
            links = _extract_job_links_from_careers_page(careers_url, r.text, limit=max_jobs_per_company)
            for link in links:
                if (time.time() - started) > hard_timeout_s:
                    break
                if len(jobs) >= max_total_jobs:
                    break
                detail = fetch_job_detail(link["url"], title_hint=link["title_hint"], timeout=8)
                jobs.append(
                    {
                        "source": "career_page",
                        "url": detail["url"],
                        "title": detail["title"],
                        "company_guess": name,
                        "location_guess": detail["location_guess"] or location,
                        "description_text": detail["description_text"],
                        "error": detail["error"],
                    }
                )
                time.sleep(delay_s)
        except Exception as exc:  # noqa: BLE001
            jobs.append(
                {
                    "source": "career_page",
                    "url": careers_url,
                    "title": "Fetch failed",
                    "company_guess": name,
                    "location_guess": location,
                    "description_text": "",
                    "error": str(exc),
                }
            )
        if i < len(companies) - 1:
            time.sleep(delay_s)
    return jobs
