import re
import time
from typing import Any, Dict
from bs4 import BeautifulSoup
import httpx

from src.logger import logging


async def scrape_single_url(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """
    Scrapes ONLY the single webpage at the provided URL.
    STRICT NON-CRAWLING: Does not follow any links or request child paths.
    Strips scripts, styles, nav, footer, headers, and ads.
    Extracts text, product information, pricing hints, and contact info.
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
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logging.error("[Single-Page Scraper] Failed to fetch %s: %s", clean_url, exc)
        return {
            "success": False,
            "url": clean_url,
            "error": f"Could not reach {clean_url}: {str(exc)}",
            "title": "",
            "text": "",
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
        len(cleaned_text),
        len(pricing_hints),
        len(contact_hints),
    )

    return {
        "success": True,
        "url": clean_url,
        "title": title,
        "text": truncated_text,
        "pricing_hints": pricing_hints,
        "contact_hints": contact_hints,
        "bestsellers": bestsellers,
    }
