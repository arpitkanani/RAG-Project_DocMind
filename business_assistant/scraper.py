import asyncio
import re
import time
from typing import Any, Dict, List
from bs4 import BeautifulSoup
import httpx

from src.logger import logging


async def scrape_single_url(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """
    Scrapes ONLY the single webpage at the provided URL.
    STRICT NON-CRAWLING: Does not follow any links or request child paths.
    Strips scripts, styles, nav, footer, headers, and ads.
    Extracts text, product information, pricing hints, and contact info.
    Returns rich diagnostic metrics (char_count, word_count, status, diagnostic_message).
    """
    start_time = time.time()
    clean_url = (url or "").strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    logging.info("[Single-Page Scraper] Initiating fetch for URL: %s", clean_url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, verify=False) as client:
            resp = await client.get(clean_url, headers=headers)
            status_code = resp.status_code
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logging.error("[Single-Page Scraper] Failed to fetch %s: %s", clean_url, exc)
        status_code = None
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            if status_code == 403:
                status = "blocked"
                diag = "HTTP 403 Forbidden: Server or anti-bot firewall (e.g. Cloudflare) denied automated scraper access."
            elif status_code == 401:
                status = "blocked"
                diag = "HTTP 401 Unauthorized: This page requires password authentication."
            elif status_code == 404:
                status = "error"
                diag = "HTTP 404 Not Found: Webpage does not exist."
            else:
                status = "error"
                diag = f"HTTP {status_code} Error: Server responded with an error."
        elif isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout)):
            status = "error"
            diag = "Connection Timeout: Server took too long to respond."
        elif isinstance(exc, httpx.ConnectError):
            status = "error"
            diag = "Connection Failed: Could not resolve domain or establish network handshake."
        else:
            status = "error"
            diag = f"Network Error: {str(exc)}"

        return {
            "success": False,
            "url": clean_url,
            "error": diag,
            "status": status,
            "status_code": status_code,
            "diagnostic_message": diag,
            "title": "",
            "text": "",
            "char_count": 0,
            "word_count": 0,
            "pricing_hints": [],
            "contact_hints": [],
            "bestsellers": [],
        }

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "iframe", "form"]):
        tag.decompose()

    # Look for common product/pricing elements
    pricing_hints = []
    price_pattern = re.compile(r"(\$|€|£|₹|USD|INR)\s?(\d+([.,]\d{1,2})?)", re.IGNORECASE)

    # Extract phone/email hints
    contact_hints = []
    phone_pattern = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

    raw_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split())

    for match in price_pattern.finditer(cleaned_text):
        snippet = cleaned_text[max(0, match.start() - 30) : min(len(cleaned_text), match.end() + 30)].strip()
        if snippet not in pricing_hints and len(pricing_hints) < 10:
            pricing_hints.append(snippet)

    for match in phone_pattern.finditer(cleaned_text):
        ph = match.group(0).strip()
        if ph not in contact_hints and len(contact_hints) < 5:
            contact_hints.append(ph)

    for match in email_pattern.finditer(cleaned_text):
        em = match.group(0).strip()
        if em not in contact_hints and len(contact_hints) < 5:
            contact_hints.append(em)

    # Check for bestseller mentions
    bestsellers = []
    bestseller_pattern = re.compile(r".{0,40}(best\s?seller|featured|popular|top\s?rated|favorite).{0,60}", re.IGNORECASE)
    for match in bestseller_pattern.finditer(cleaned_text):
        bs_snippet = match.group(0).strip().replace("\n", " ")
        if bs_snippet not in bestsellers and len(bestsellers) < 5:
            bestsellers.append(bs_snippet)

    # Cap cleaned text length for prompt context
    truncated_text = cleaned_text[:12000]

    duration = time.time() - start_time
    logging.info(
        "[Single-Page Scraper] Successfully scraped %s in %.2fs (Chars: %d, Prices: %d, Contacts: %d)",
        clean_url,
        duration,
        char_count,
        len(pricing_hints),
        len(contact_hints),
    )

    if char_count < 250:
        status = "low_content"
        diag = f"Low text yield ({char_count} chars). The site may be a client-side rendered Single Page Application (SPA)."
    else:
        status = "success"
        diag = f"Successfully extracted {char_count:,} characters ({word_count:,} words)."

    return {
        "success": True,
        "url": clean_url,
        "title": title,
        "text": truncated_text,
        "char_count": char_count,
        "word_count": word_count,
        "status": status,
        "status_code": 200,
        "diagnostic_message": diag,
        "pricing_hints": pricing_hints,
        "contact_hints": contact_hints,
        "bestsellers": bestsellers,
    }


async def scrape_multiple_urls(urls: List[str], timeout: float = 15.0) -> Dict[str, Any]:
    """
    Scrapes multiple URLs concurrently (up to 5 pages), aggregating extracted text,
    pricing, contact points, and returning granular per-page health diagnostics.
    """
    cleaned_urls: List[str] = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            # Support comma or newline delimited strings in any input
            for sub_u in re.split(r"[\n,]+", u):
                clean = sub_u.strip()
                if clean and clean not in cleaned_urls:
                    cleaned_urls.append(clean)

    cleaned_urls = cleaned_urls[:5]  # Limit to 5 pages to respect timeout and rate limits
    if not cleaned_urls:
        return {
            "success": False,
            "url": "",
            "title": "",
            "text": "",
            "char_count": 0,
            "word_count": 0,
            "pricing_hints": [],
            "contact_hints": [],
            "bestsellers": [],
            "pages": [],
            "overall_status": "error",
            "diagnostic_message": "Please enter at least one valid website URL to scrape.",
        }

    tasks = [scrape_single_url(u, timeout=timeout) for u in cleaned_urls]
    pages: List[Dict[str, Any]] = await asyncio.gather(*tasks)

    total_chars = sum(p.get("char_count", 0) for p in pages)
    total_words = sum(p.get("word_count", 0) for p in pages)

    combined_text_blocks = []
    pricing_hints: List[str] = []
    contact_hints: List[str] = []
    bestsellers: List[str] = []
    titles: List[str] = []

    success_pages = [p for p in pages if p.get("success")]
    for p in success_pages:
        if p.get("title") and p["title"] not in titles:
            titles.append(p["title"])
        for pr in p.get("pricing_hints", []):
            if pr not in pricing_hints:
                pricing_hints.append(pr)
        for c in p.get("contact_hints", []):
            if c not in contact_hints:
                contact_hints.append(c)
        for b in p.get("bestsellers", []):
            if b not in bestsellers:
                bestsellers.append(b)

        if p.get("text"):
            header = f"=== WEBPAGE: {p.get('url')} (Title: {p.get('title') or 'Untitled'}) ==="
            combined_text_blocks.append(f"{header}\n{p['text']}")

    combined_text = "\n\n".join(combined_text_blocks)[:16000]

    primary_title = " | ".join(titles[:2]) if titles else (pages[0].get("title") or "")
    primary_url = ", ".join(cleaned_urls)

    if total_chars >= 500:
        overall_status = "high_yield"
        diag = f"High content yield: Extracted {total_chars:,} characters ({total_words:,} words) across {len(success_pages)} of {len(pages)} page(s)."
    elif total_chars >= 150:
        overall_status = "moderate_yield"
        diag = f"Moderate content yield: Extracted {total_chars:,} characters across {len(success_pages)} page(s)."
    elif any(p.get("status") == "blocked" for p in pages):
        overall_status = "blocked"
        diag = "Website access was blocked by anti-bot firewall (HTTP 403 / Cloudflare). Automated crawlers cannot access this domain."
    elif total_chars > 0:
        overall_status = "low_content"
        diag = f"Low Content Warning: Only {total_chars} characters extracted. The site may be a client-side rendered Single Page Application (SPA)."
    else:
        overall_status = "error"
        diag = "Could not extract content from the provided URL(s)."

    return {
        "success": bool(success_pages and total_chars > 0),
        "url": primary_url,
        "title": primary_title,
        "text": combined_text,
        "char_count": total_chars,
        "word_count": total_words,
        "pricing_hints": pricing_hints[:12],
        "contact_hints": contact_hints[:6],
        "bestsellers": bestsellers[:6],
        "pages": pages,
        "overall_status": overall_status,
        "diagnostic_message": diag,
    }
