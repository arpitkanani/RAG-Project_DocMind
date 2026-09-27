import sys
from typing import Any, List, Optional

from langchain_core.tools import tool #type: ignore
from langsmith import traceable #type: ignore
import yaml

from src.chains.qa_chain import (
    build_citations,
    build_source_only_citations,
    format_docs,
    merge_same_location_docs,
)
from src.components.retriever import Retriever
from src.exception import CollectionNotFoundError, CustomException, KnowledgeBaseEmptyError
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


@tool
@traceable(run_type="tool", name="rag_query")
async def rag_query(
    query: str,
    collection_names: Optional[List[str]] = None,
    session_id: str = "default",
    user_id: str = "default",
) -> str:
    """Search the uploaded documents and knowledge base for relevant facts.

    Args:
        query: Specific question to look up in the documents.
        collection_names: Collections to scope the search to.
        session_id: Current user session ID.
        user_id: Current user ID.

    Returns:
        Retrieved document excerpts with source citations.
    """
    try:
        logging.info(
            "rag_query tool | query: %r | scope: %s | session: %s",
            query, collection_names, session_id
        )

        retriever = Retriever(collection_names=collection_names)
        ranked_docs = await retriever.retrieve_ranked(query)
        docs = [doc for doc, _, _ in ranked_docs]
        docs = merge_same_location_docs(docs)

        if not docs:
            return "No relevant information found in the documents for this query."

        formatted = format_docs(docs)
        citations = build_citations(docs)

        logging.info("rag_query retrieved %d chunks", len(docs))

        # Return formatted content with citations
        return f"{formatted}\n\n{citations}" if citations else formatted

    except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
        logging.warning("rag_query collection error: %s", e)
        return "The document collection is empty or was not found."
    except Exception as e:
        logging.exception("rag_query tool failed")
        return f"Error querying documents: {str(e)}"


@tool
@traceable(run_type="tool", name="summarize_document")
async def summarize_document(
    collection_names: Optional[List[str]] = None,
    session_id: str = "default",
    user_id: str = "default",
) -> str:
    """Retrieve full document context for summarization.

    Args:
        collection_names: Collections to summarize.
        session_id: Current user session ID.
        user_id: Current user ID.

    Returns:
        Full document text with source citations.
    """
    try:
        logging.info(
            "summarize_document tool | scope: %s | session: %s",
            collection_names, session_id
        )

        retriever = Retriever(collection_names=collection_names)
        max_chars = config.get("retriever", {}).get("summary_max_chars", 6000)
        docs = await retriever.get_full_context(max_chars=max_chars)
        docs = merge_same_location_docs(docs)

        if not docs:
            return "No documents found to summarize."

        formatted = format_docs(docs)
        citations = build_source_only_citations(docs)

        logging.info("summarize_document retrieved %d chunks", len(docs))
        return f"{formatted}\n\n{citations}" if citations else formatted

    except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
        logging.warning("summarize_document collection error: %s", e)
        return "The document collection is empty or was not found."
    except Exception as e:
        logging.exception("summarize_document tool failed")
        return f"Error retrieving document context: {str(e)}"


AVAILABLE_TOOLS = [rag_query, summarize_document]


# ══════════════════════════════════════════════════════════════════════════════
# Non-Document Conversational (Chitchat) ReAct Tools
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import requests
import httpx
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# 1. Search tool for live web information and current events
@tool
@traceable(name="search_tool")
def search_tool(query: str) -> str:
    """Search the web for current general news, public events, and factual web content.
    CRITICAL: Do NOT use this tool for stock prices, financial market quotes, weather forecasts, or math calculations!
    Use 'get_stock_price' for stock prices, 'get_weather' for weather, and 'calculator' for math computations."""
    q = str(query).strip()
    if not q:
        return "No search query provided."

    # Strategy 1: Try duckduckgo_search library directly (more up-to-date and configurable)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Try text search with backend fallbacks
            results = None
            for backend in ("api", "html", "lite"):
                try:
                    results = list(ddgs.text(q, max_results=5, backend=backend))
                    if results:
                        break
                except Exception as b_err:
                    logging.debug("DDGS backend %s failed: %s", backend, b_err)
                    continue

            if results:
                formatted_snippets = []
                for r in results:
                    title = r.get("title", "").strip()
                    body = r.get("body", "") or r.get("snippet", "")
                    href = r.get("href", "") or r.get("link", "")
                    entry = f"• {title}: {body}"
                    if href:
                        entry += f" (Source: {href})"
                    formatted_snippets.append(entry)
                out = "\n\n".join(formatted_snippets)
                logging.info("search_tool returned %d results via duckduckgo_search", len(results))
                return out
    except Exception as e1:
        logging.warning("search_tool: duckduckgo_search direct lookup failed: %s", e1)

    # Strategy 2: Fallback to langchain_community DuckDuckGoSearchResults / DuckDuckGoSearchRun
    try:
        from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
        try:
            ddg_results = DuckDuckGoSearchResults(max_results=5)
            res_str = ddg_results.invoke(q)
            if res_str and "No good DuckDuckGo Search Result was found" not in res_str:
                return res_str
        except Exception:
            pass

        ddg_run = DuckDuckGoSearchRun()
        res_str = ddg_run.invoke(q)
        if res_str and "No good DuckDuckGo Search Result was found" not in res_str:
            return res_str
    except Exception as e2:
        logging.warning("search_tool: langchain DuckDuckGo fallback failed: %s", e2)

    # Strategy 3: Fast Wikipedia summary API fallback for encyclopedic/factual queries
    try:
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(q)}"
        headers = {"User-Agent": "DocuVortex-AI/2.0 (assistant@docuvortex.local)"}
        resp = requests.get(wiki_url, headers=headers, timeout=6)
        if resp.status_code == 200:
            wdata = resp.json()
            extract = wdata.get("extract")
            if extract:
                title = wdata.get("title", q)
                logging.info("search_tool: retrieved Wikipedia summary for %s", title)
                return f"Summary from Wikipedia ({title}): {extract}"
    except Exception as e3:
        logging.debug("search_tool: Wikipedia fallback error: %s", e3)

    return f"Live search could not locate specific real-time results for '{q}'. Please proceed based on your knowledge base."


@tool
@traceable(name="calculator")
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers: add, sub, mul, div.
    CRITICAL: ALWAYS use this tool for arithmetic expressions and math calculations (e.g. 230*460). Never use search_tool for math."""
    try:
        n1 = float(first_num)
        n2 = float(second_num)
        op = str(operation).lower().strip()
        if op in ("add", "+", "addition"):
            return {"result": n1 + n2}
        elif op in ("sub", "-", "subtract", "subtraction"):
            return {"result": n1 - n2}
        elif op in ("mul", "*", "multiply", "multiplication"):
            return {"result": n1 * n2}
        elif op in ("div", "/", "divide", "division"):
            if n2 == 0:
                return {"error": "Division by zero is not allowed"}
            return {"result": n1 / n2}
        return {"error": f"Unsupported operation '{operation}'. Use add, sub, mul, or div."}
    except Exception as e:
        return {"error": str(e)}


TICKER_MAP = {
    # US Tech Giants & Semis
    "APPLE": "AAPL",
    "AAPL": "AAPL",
    "MICROSOFT": "MSFT",
    "MSFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "GOOGL": "GOOGL",
    "GOOG": "GOOG",
    "AMAZON": "AMZN",
    "AMZN": "AMZN",
    "TESLA": "TSLA",
    "TSLA": "TSLA",
    "NVIDIA": "NVDA",
    "NVDA": "NVDA",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
    "NFLX": "NFLX",
    "AMD": "AMD",
    "ADVANCED MICRO DEVICES": "AMD",
    "INTEL": "INTC",
    "INTC": "INTC",
    "IBM": "IBM",
    "ORACLE": "ORCL",
    "ORCL": "ORCL",
    "SALESFORCE": "CRM",
    "CRM": "CRM",
    "ADOBE": "ADBE",
    "ADBE": "ADBE",
    "UBER": "UBER",
    "AIRBNB": "ABNB",
    "PALANTIR": "PLTR",
    "PLTR": "PLTR",
    "QUALCOMM": "QCOM",
    "QCOM": "QCOM",
    "BROADCOM": "AVGO",
    "AVGO": "AVGO",
    "CISCO": "CSCO",
    "CSCO": "CSCO",
    # Consumer & Finance
    "DISNEY": "DIS",
    "DIS": "DIS",
    "COCA COLA": "KO",
    "COCA-COLA": "KO",
    "COKE": "KO",
    "KO": "KO",
    "PEPSI": "PEP",
    "PEPSICO": "PEP",
    "PEP": "PEP",
    "STARBUCKS": "SBUX",
    "SBUX": "SBUX",
    "MCDONALDS": "MCD",
    "MCDONALD'S": "MCD",
    "MCD": "MCD",
    "BOEING": "BA",
    "BA": "BA",
    "WALMART": "WMT",
    "WMT": "WMT",
    "TARGET": "TGT",
    "TGT": "TGT",
    "COSTCO": "COST",
    "COST": "COST",
    "NIKE": "NKE",
    "NKE": "NKE",
    "BERKSHIRE": "BRK-B",
    "BERKSHIRE HATHAWAY": "BRK-B",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "JPM": "JPM",
    "GOLDMAN SACHS": "GS",
    "GS": "GS",
    "MORGAN STANLEY": "MS",
    "MS": "MS",
    "VISA": "V",
    "V": "V",
    "MASTERCARD": "MA",
    "MA": "MA",
    "SPOTIFY": "SPOT",
    "SPOT": "SPOT",
    "COINBASE": "COIN",
    "COIN": "COIN",
    # Crypto
    "BITCOIN": "BTC-USD",
    "BTC": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "ETH": "ETH-USD",
    "SOLANA": "SOL-USD",
    "SOL": "SOL-USD",
    "DOGECOIN": "DOGE-USD",
    "DOGE": "DOGE-USD",
    # Indices & ETFs
    "SP500": "SPY",
    "S&P 500": "SPY",
    "S&P500": "SPY",
    "SPY": "SPY",
    "NASDAQ": "QQQ",
    "QQQ": "QQQ",
    "DOW": "DIA",
    "DOW JONES": "DIA",
    "DIA": "DIA",
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "BSE SENSEX": "^BSESN",
    # Indian / Global Equities (NSE/BSE)
    "MAHINDRA": "M&M.NS",
    "MAHINDRA & MAHINDRA": "M&M.NS",
    "MAHINDRA AND MAHINDRA": "M&M.NS",
    "M&M": "M&M.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATA MOTOR": "TATAMOTORS.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "RIL": "RELIANCE.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "SBI": "SBIN.NS",
    "SBIN": "SBIN.NS",
    "AXIS BANK": "AXISBANK.NS",
    "AXIS": "AXISBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAK": "KOTAKBANK.NS",
    "KOTAK BANK": "KOTAKBANK.NS",
    "KOTAK MAHINDRA BANK": "KOTAKBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "ITC": "ITC.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "LARSEN & TOUBRO": "LT.NS",
    "LARSEN": "LT.NS",
    "L&T": "LT.NS",
    "LT": "LT.NS",
    "WIPRO": "WIPRO.NS",
    "HCL": "HCLTECH.NS",
    "HCL TECH": "HCLTECH.NS",
    "HCL TECHNOLOGIES": "HCLTECH.NS",
    "MARUTI": "MARUTI.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "ADANI": "ADANIENT.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "SWIGGY": "SWIGGY.NS",
    "JIO FINANCIAL": "JIOFIN.NS",
    "JIOFIN": "JIOFIN.NS",
}


def _search_yahoo_ticker(query: str) -> str:
    """Use Yahoo Finance search API to dynamically resolve any unmapped global company name."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}&quotesCount=3&newsCount=0"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("quotes", [])
            for q in quotes:
                sym = q.get("symbol")
                quote_type = q.get("quoteType", "")
                if sym and quote_type in ("EQUITY", "ETF", "CRYPTOCURRENCY", "INDEX", "MUTUALFUND"):
                    logging.info("_search_yahoo_ticker: resolved '%s' -> %s (%s)", query, sym, q.get("shortname", ""))
                    return sym
            if quotes and quotes[0].get("symbol"):
                return quotes[0]["symbol"]
    except Exception as e:
        logging.debug("_search_yahoo_ticker failed for %s: %s", query, e)
    return ""


def _resolve_stock_ticker(raw_symbol: str) -> str:
    """Normalize input company name or ticker symbol to a standard financial ticker."""
    cleaned = str(raw_symbol or "").strip().upper().replace("$", "")
    # Remove common conversational or corporate suffixes
    cleaned = re.sub(
        r"\b(STOCK|SHARE|SHARES|PRICE|PRICES|QUOTE|QUOTES|TICKER|INC|CORP|LTD|COMPANY|THE|OF|FOR)\b",
        "",
        cleaned,
    ).strip()

    # 1. Exact match in comprehensive TICKER_MAP
    if cleaned in TICKER_MAP:
        return TICKER_MAP[cleaned]

    # 2. Match multi-word / word-boundary company names in TICKER_MAP (longest names first)
    for name in sorted(TICKER_MAP.keys(), key=len, reverse=True):
        if len(name) >= 3 and re.search(rf"\b{re.escape(name)}\b", cleaned):
            return TICKER_MAP[name]

    # 3. Dynamic lookup via Yahoo Finance search endpoint
    dynamic_sym = _search_yahoo_ticker(cleaned)
    if dynamic_sym:
        return dynamic_sym

    return cleaned if cleaned else "AAPL"


@tool
@traceable(name="get_stock_price")
def get_stock_price(symbol: str = "", ticker: str = "", company: str = "") -> dict:
    """Fetch latest real-time stock price, day change, day high/low, volume, and market quote for ANY company or stock ticker symbol worldwide.
    Examples: Tesla (TSLA), Apple (AAPL), Mahindra & Mahindra (M&M), Axis Bank (AXISBANK), Tata Motors (TATAMOTORS), Reliance, Microsoft, Nvidia, Google, Bitcoin, etc.
    CRITICAL: ALWAYS use this tool for ANY question about stock prices, share prices, market quotes, or tickers. Never use search_tool for stock prices."""
    clean_input = str(symbol or ticker or company or "").strip()
    if not clean_input:
        clean_input = "AAPL"
    ticker_sym = _resolve_stock_ticker(clean_input)

    # Strategy 1: Yahoo Finance Real-Time Chart API (query1 and query2 endpoints)
    for yahoo_host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            logging.info("get_stock_price: querying %s for %s (ticker: %s)", yahoo_host, clean_input, ticker_sym)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://finance.yahoo.com",
            }
            url = f"https://{yahoo_host}/v8/finance/chart/{ticker_sym}?interval=1d&range=1d"
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200:
                data = res.json()
                results = data.get("chart", {}).get("result")
                if results and len(results) > 0:
                    meta = results[0].get("meta", {})
                    price = meta.get("regularMarketPrice")
                    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
                    currency = meta.get("currency", "USD")
                    high = meta.get("regularMarketDayHigh")
                    low = meta.get("regularMarketDayLow")
                    volume = meta.get("regularMarketVolume")
                    exchange = meta.get("exchangeName") or meta.get("fullExchangeName")

                    if price is not None:
                        change = (price - prev_close) if prev_close else 0.0
                        change_pct = ((change / prev_close) * 100) if prev_close else 0.0
                        return {
                            "symbol": ticker_sym,
                            "company_queried": clean_input,
                            "current_price": f"{price:.2f} {currency}",
                            "change": f"{change:+.2f} ({change_pct:+.2f}%)",
                            "previous_close": f"{prev_close:.2f} {currency}" if prev_close else "N/A",
                            "day_high": f"{high:.2f} {currency}" if high else "N/A",
                            "day_low": f"{low:.2f} {currency}" if low else "N/A",
                            "volume": f"{volume:,}" if volume else "N/A",
                            "exchange": exchange or "US",
                            "source": "Yahoo Finance (Real-Time)",
                            "status": "success",
                        }
        except Exception as e1:
            logging.warning("get_stock_price: %s API failed for %s: %s", yahoo_host, ticker_sym, e1)

    # Strategy 2: Alpha Vantage fallback
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "YBAMDXZOUTYN8L4J")
        av_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_sym}&apikey={api_key}"
        res = requests.get(av_url, timeout=8)
        av_data = res.json()
        if "Global Quote" in av_data and av_data["Global Quote"]:
            quote = av_data["Global Quote"]
            price_str = quote.get("05. price")
            if price_str:
                return {
                    "symbol": quote.get("01. symbol", ticker_sym),
                    "company_queried": clean_input,
                    "current_price": f"{float(price_str):.2f} USD",
                    "change": f"{quote.get('09. change', '0')} ({quote.get('10. change percent', '0%')})",
                    "day_high": quote.get("03. high", "N/A"),
                    "day_low": quote.get("04. low", "N/A"),
                    "volume": quote.get("06. volume", "N/A"),
                    "latest_trading_day": quote.get("07. latest trading day", "N/A"),
                    "source": "Alpha Vantage",
                    "status": "success",
                }
    except Exception as e2:
        logging.warning("get_stock_price: Alpha Vantage lookup failed for %s: %s", ticker_sym, e2)

    # Strategy 3: DuckDuckGo / web search fallback
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{ticker_sym} stock price quote market today", max_results=3))
            if results:
                snippets = [f"{r.get('title')}: {r.get('body')}" for r in results]
                joined_snippets = "\n".join(snippets)
                # Attempt to extract price from snippets
                price_match = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*(?:USD|dollars)", joined_snippets, re.IGNORECASE)
                extracted_price = f"${price_match.group(1) or price_match.group(2)} USD" if price_match else "Market Quote (Live)"

                return {
                    "symbol": ticker_sym,
                    "company_queried": clean_input,
                    "current_price": extracted_price,
                    "change": "N/A",
                    "day_high": "N/A",
                    "day_low": "N/A",
                    "exchange": "US",
                    "live_search_quotes": joined_snippets,
                    "source": "Live Web Search (DuckDuckGo)",
                    "status": "success",
                }
    except Exception as e3:
        logging.warning("get_stock_price: Search fallback failed: %s", e3)

    return {
        "symbol": ticker_sym,
        "company_queried": clean_input,
        "current_price": "Quote Unavailable",
        "message": f"Unable to fetch live quote for {ticker_sym} at this moment. The markets may be closed or quote servers unreachable.",
        "status": "completed",
    }


@tool
@traceable(name="get_weather")
async def get_weather(location: str = "", city: str = "") -> str:
    """Get live weather conditions, current temperature, humidity, wind, and description for ANY city or location worldwide.
    Examples: 'Mumbai', 'Delhi', 'New York', 'London', 'Tokyo', 'San Francisco', etc.
    CRITICAL: ALWAYS use this tool for ANY question asking about weather, temperature, rain, or climate. Never use search_tool for weather queries."""
    clean_loc = str(location or city or "").strip()
    if not clean_loc:
        return "Please specify a location to get the weather for."

    # Strategy 1: OpenWeatherMap API if key is available
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={requests.utils.quote(clean_loc)}&appid={api_key}&units=metric"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    main = data.get("main", {})
                    temp = main.get("temp")
                    humidity = main.get("humidity")
                    weather_list = data.get("weather", [])
                    desc = weather_list[0].get("description", "clear") if weather_list else "clear"
                    city_name = data.get("name", clean_loc)
                    country = data.get("sys", {}).get("country", "")
                    loc_label = f"{city_name}, {country}" if country else city_name
                    return f"Weather in {loc_label}: {temp}°C, {desc.capitalize()}, Humidity: {humidity}%"
        except Exception as e:
            logging.warning("get_weather OpenWeatherMap attempt failed for %s: %s", clean_loc, e)

    # Strategy 2: Resilient free wttr.in fallback (no API key required, reliable worldwide)
    try:
        wttr_url = f"https://wttr.in/{requests.utils.quote(clean_loc)}?format=j1"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(wttr_url)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current_condition", [{}])[0]
                temp_c = current.get("temp_C", "N/A")
                feels_like = current.get("FeelsLikeC", "N/A")
                humidity = current.get("humidity", "N/A")
                desc_obj = current.get("weatherDesc", [{}])[0]
                desc = desc_obj.get("value", "Clear")
                wind = current.get("windspeedKmph", "N/A")
                nearest = data.get("nearest_area", [{}])[0]
                area_name = nearest.get("areaName", [{}])[0].get("value", clean_loc)
                country_name = nearest.get("country", [{}])[0].get("value", "")
                loc_label = f"{area_name}, {country_name}" if country_name else area_name

                return (
                    f"Weather in {loc_label}: {temp_c}°C (Feels like {feels_like}°C), "
                    f"{desc}, Humidity: {humidity}%, Wind: {wind} km/h"
                )
    except Exception as e2:
        logging.warning("get_weather wttr.in fallback failed for %s: %s", clean_loc, e2)

    return f"Could not retrieve live weather information for '{clean_loc}' at this moment. Please check the city name and try again."


@tool
@traceable(name="search_arxiv")
async def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search the arXiv preprint database for scientific papers, academic studies, or research articles.
    
    Args:
        query: Topic, keywords, or paper title to search for.
        max_results: Maximum number of papers to return (default: 3).
        
    Returns:
        Titles, links, and summaries of matching research papers separated by '---'.
    """
    clean_q = str(query).strip()
    if not clean_q:
        return "Please provide a search topic or keywords for arXiv."

    max_r = max(1, min(int(max_results), 10))
    url = f"http://export.arxiv.org/api/query?search_query=all:{clean_q}&start=0&max_results={max_r}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return f"arXiv search failed with HTTP status {resp.status_code}."

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"No research papers found on arXiv for query '{clean_q}'."

        formatted_papers = []
        for entry in entries:
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            link = entry.findtext("atom:id", "", ns).strip()
            summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            if len(summary) > 300:
                summary = summary[:300].rstrip() + "..."

            formatted_papers.append(
                f"📄 **{title}**\nLink: {link}\nSummary: {summary}"
            )

        return "\n\n---\n\n".join(formatted_papers)
    except httpx.RequestError as e:
        logging.warning("search_arxiv network error: %s", e)
        return f"Network error searching arXiv for '{clean_q}'."
    except Exception as e:
        logging.error("search_arxiv failed for %s: %s", clean_q, e)
        return f"Could not complete arXiv search for '{clean_q}'."


chitchat_tools = [search_tool, calculator, get_stock_price, get_weather, search_arxiv]
