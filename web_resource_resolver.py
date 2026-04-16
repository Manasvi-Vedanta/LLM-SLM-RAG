"""
web_resource_resolver.py – Web Resource Resolver
--------------------------------------------------
Takes resource names from a learning path session (e.g. "Real Python Tutorials",
"Codecademy Python Course"), resolves them to actual URLs via web search, scrapes
readable content, and returns LangChain Documents for ingestion.

Handles failures gracefully — if a resource can't be resolved or scraped,
logs a warning and moves on.

Search strategy (in order of preference):
  1. Brave Search (scrape search results page)
  2. DuckDuckGo API (`duckduckgo_search` / `ddgs` library)
  3. DuckDuckGo HTML fallback (legacy, often blocked)

Content extraction:
  1. trafilatura (best boilerplate removal)
  2. BeautifulSoup fallback (manual extraction)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse, unquote

from langchain_core.documents import Document

import config
import transcript_extractor

logger = logging.getLogger(__name__)

# Domains to skip in search results
_SKIP_DOMAINS = frozenset({
    "brave.com", "bing.com", "google.com", "microsoft.com",
    "duckduckgo.com", "yahoo.com", "ad.doubleclick.net",
    "facebook.com", "x.com", "twitter.com", "instagram.com",
    "linkedin.com", "tiktok.com", "pinterest.com",
})


# ──────────────────────────────────────────────
# URL resolution via web search
# ──────────────────────────────────────────────

def _resolve_via_brave(query: str) -> Optional[str]:
    """Search via Brave Search (most reliable, no API key needed)."""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://search.brave.com/search"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        resp = requests.get(url, params=params, headers=headers,
                            timeout=config.WEB_SCRAPE_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract result URLs from Brave's HTML
        seen_domains: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                continue
            parsed = urlparse(href)
            domain = parsed.netloc.lower()

            if any(skip in domain for skip in _SKIP_DOMAINS):
                continue
            if domain in seen_domains:
                continue
            if len(parsed.path) < 1:
                continue

            seen_domains.add(domain)
            logger.info("Brave resolved '%s' -> %s", query[:50], href)
            return href

        logger.debug("Brave: no results for '%s'", query[:50])
        return None

    except Exception as exc:
        logger.warning("Brave search failed for '%s': %s", query[:50], exc)
        return None


def _resolve_via_ddgs(query: str) -> Optional[str]:
    """Search via duckduckgo_search / ddgs library."""
    try:
        # Try the newer 'ddgs' package first, fall back to 'duckduckgo_search'
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = list(DDGS().text(query, max_results=3))
        for r in results:
            href = r.get("href", "")
            if href:
                parsed = urlparse(href)
                if not any(skip in parsed.netloc for skip in _SKIP_DOMAINS):
                    logger.info("DDGS resolved '%s' -> %s", query[:50], href)
                    return href

        logger.debug("DDGS: no results for '%s'", query[:50])
        return None

    except ImportError:
        logger.debug("DDGS library not installed")
        return None
    except Exception as exc:
        logger.debug("DDGS search failed for '%s': %s", query[:50], exc)
        return None


def _resolve_via_ddg_html(query: str) -> Optional[str]:
    """Legacy: scrape DuckDuckGo HTML search page (often blocked)."""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        }

        resp = requests.post(url, data=params, headers=headers,
                             timeout=config.WEB_SCRAPE_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for link in soup.select("a.result__a"):
            href = link.get("href", "")
            if href and href.startswith("http"):
                parsed = urlparse(href)
                if not any(d in parsed.netloc for d in _SKIP_DOMAINS):
                    logger.info("DDG HTML resolved '%s' -> %s", query[:50], href)
                    return href

        logger.debug("DDG HTML: no results for '%s'", query[:50])
        return None

    except Exception as exc:
        logger.debug("DDG HTML search failed for '%s': %s", query[:50], exc)
        return None


def resolve_resource(name: str, skill_label: str) -> Optional[str]:
    """Search the web for a resource name and return the best URL.

    Tries Brave Search first, then DDGS library, then DDG HTML fallback.

    Parameters
    ----------
    name : str
        The resource name, e.g. "Real Python Tutorials".
    skill_label : str
        The skill context to improve search relevance.

    Returns
    -------
    str or None
        A URL, or None if resolution failed.
    """
    query = f"{name} {skill_label} tutorial"

    # Strategy 1: DDGS library (most reliable, no rate limits)
    url = _resolve_via_ddgs(query)
    if url:
        return url

    # Strategy 2: Brave Search (rate-limits after a few requests)
    url = _resolve_via_brave(query)
    if url:
        return url

    # Strategy 3: DDG HTML (legacy fallback)
    url = _resolve_via_ddg_html(query)
    if url:
        return url

    logger.warning("Could not resolve resource: '%s' (all search backends failed)", name)
    return None


# ──────────────────────────────────────────────
# Content scraping
# ──────────────────────────────────────────────

def _scrape_with_trafilatura(url: str) -> Optional[str]:
    """Extract readable text using trafilatura (best boilerplate removal)."""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        text = trafilatura.extract(downloaded, include_comments=False,
                                   include_tables=True)
        if text and len(text) >= 200:
            logger.debug("trafilatura extracted %d chars from %s", len(text), url)
            return text

        return None

    except ImportError:
        return None
    except Exception as exc:
        logger.debug("trafilatura failed for %s: %s", url, exc)
        return None


def _scrape_with_bs4(url: str) -> Optional[str]:
    """Fallback: extract readable text using BeautifulSoup."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        resp = requests.get(url, headers=headers,
                            timeout=config.WEB_SCRAPE_TIMEOUT)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type.lower() and "text" not in content_type.lower():
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "iframe", "noscript",
                         "button", "input", "select"]):
            tag.decompose()

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"class": re.compile(r"content|article|post|entry", re.I)})
            or soup.find("div", {"id": re.compile(r"content|article|post|entry", re.I)})
            or soup.find("body")
        )

        if not main_content:
            return None

        text = main_content.get_text(separator="\n", strip=True)

        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 20:
                lines.append(line)

        cleaned = "\n".join(lines)

        if len(cleaned) < 200:
            return None

        return cleaned

    except Exception as exc:
        logger.debug("BS4 scrape failed for %s: %s", url, exc)
        return None


def scrape_content(url: str) -> Optional[str]:
    """Fetch a web page and extract readable text content.

    Tries trafilatura first (superior boilerplate removal), then falls
    back to BeautifulSoup.

    Returns
    -------
    str or None
        Readable text, or None if scraping failed.
    """
    # Strategy 1: trafilatura
    text = _scrape_with_trafilatura(url)
    if text:
        # Cap at ~50k chars
        if len(text) > 50000:
            text = text[:50000]
        logger.info("Scraped %d chars from %s (trafilatura)", len(text), url)
        return text

    # Strategy 2: BeautifulSoup fallback
    text = _scrape_with_bs4(url)
    if text:
        if len(text) > 50000:
            text = text[:50000]
        logger.info("Scraped %d chars from %s (bs4)", len(text), url)
        return text

    logger.warning("Failed to scrape %s (both extractors failed)", url)
    return None


# ──────────────────────────────────────────────
# Session-level resolver
# ──────────────────────────────────────────────

def resolve_and_scrape_session_resources(
    resources: dict,
    skill_label: str,
) -> tuple[list[Document], list[dict]]:
    """Resolve and scrape all resources for a session.

    Video URLs (YouTube, Vimeo, etc.) are identified during resolution and
    returned separately so the caller can route them to transcript extraction.

    Parameters
    ----------
    resources : dict
        The ``learning_resources`` dict from a session, with keys like
        ``online_courses``, ``tutorials``, ``documentation``, etc.
    skill_label : str
        Primary skill for the session (used for search context).

    Returns
    -------
    tuple[list[Document], list[dict]]
        - LangChain Documents with scraped web content and metadata.
        - List of video resource dicts: ``{"name", "url", "category"}``.
    """
    # Collect all unique resource names across categories
    seen_names: set[str] = set()
    resource_list: list[tuple[str, str]] = []  # (name, category)

    # Priority order: courses > tutorials > documentation > others
    priority_order = [
        "online_courses", "tutorials", "documentation",
        "practice_platforms", "communities"
    ]

    for category in priority_order:
        names = resources.get(category, [])
        if not names:
            continue
        for name in names:
            name_lower = name.strip().lower()
            if name_lower and name_lower not in seen_names:
                seen_names.add(name_lower)
                resource_list.append((name.strip(), category))

    # Cap at MAX_RESOURCES_PER_SESSION
    resource_list = resource_list[:config.MAX_RESOURCES_PER_SESSION]

    if not resource_list:
        logger.info("No resources to resolve for skill '%s'", skill_label)
        return [], []

    logger.info("Resolving %d resources for skill '%s'", len(resource_list), skill_label)

    documents: list[Document] = []
    video_resources: list[dict] = []
    seen_urls: set[str] = set()

    for i, (name, category) in enumerate(resource_list):
        # Delay between requests to avoid Brave rate limiting (429)
        if i > 0:
            time.sleep(3.0)

        # Resolve name to URL
        url = resolve_resource(name, skill_label)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # Check if this is a video URL — route separately instead of scraping
        if transcript_extractor.is_video_url(url):
            logger.info("Video URL detected for '%s' -> %s (routed to transcript extraction)", name, url)
            video_resources.append({
                "name": name,
                "url": url,
                "category": category,
            })
            continue

        # Scrape content (non-video URLs only)
        content = scrape_content(url)
        if not content:
            continue

        doc = Document(
            page_content=content,
            metadata={
                "source_type": "web",
                "resource_name": name,
                "resource_category": category,
                "url": url,
                "skill_label": skill_label,
            },
        )
        documents.append(doc)

    logger.info("Successfully scraped %d/%d resources for '%s' (%d video URLs found)",
                len(documents), len(resource_list), skill_label, len(video_resources))

    return documents, video_resources
