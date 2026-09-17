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
    """Search the web for current events, latest news, recent facts, and live information."""
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
    """Perform a basic arithmetic operation on two numbers: add, sub, mul, div."""
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


@tool
@traceable(name="get_stock_price")
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price and market quote for a given stock ticker symbol (e.g. 'AAPL', 'TSLA', 'MSFT', 'NVDA') using Alpha Vantage."""
    clean_symbol = str(symbol).strip().upper()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "YBAMDXZOUTYN8L4J")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={clean_symbol}&apikey={api_key}"
    try:
        logging.info("get_stock_price: querying symbol %s", clean_symbol)
        res = requests.get(url, timeout=10)
        data = res.json()
        if "Global Quote" in data:
            quote = data["Global Quote"]
            if quote:
                return {
                    "symbol": quote.get("01. symbol", clean_symbol),
                    "price": quote.get("05. price"),
                    "change": quote.get("09. change"),
                    "change_percent": quote.get("10. change percent"),
                    "high": quote.get("03. high"),
                    "low": quote.get("04. low"),
                    "volume": quote.get("06. volume"),
                    "latest_trading_day": quote.get("07. latest trading day"),
                }
        return data
    except Exception as e:
        logging.error("get_stock_price error for %s: %s", clean_symbol, e)
        return {"error": str(e)}


@tool
@traceable(name="get_weather")
async def get_weather(location: str) -> str:
    """Get current weather details for a specific city or geographic location.
    
    Args:
        location: City or location name, e.g. 'London', 'Mumbai', 'New York'.
        
    Returns:
        Formatted temperature in Celsius, description, and humidity.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather service is currently unavailable (OPENWEATHER_API_KEY is not configured)."

    clean_loc = str(location).strip()
    if not clean_loc:
        return "Please specify a location to get the weather for."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={clean_loc}&appid={api_key}&units=metric"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return f"Could not find weather data for '{clean_loc}'. Please check the city spelling."
            if resp.status_code != 200:
                return f"Unable to fetch weather for '{clean_loc}' (Error: HTTP {resp.status_code})."

            data = resp.json()
            main = data.get("main", {})
            temp = main.get("temp")
            humidity = main.get("humidity")
            weather_list = data.get("weather", [])
            desc = weather_list[0].get("description", "clear") if weather_list else "clear"
            city_name = data.get("name", clean_loc)
            sys_info = data.get("sys", {})
            country = sys_info.get("country", "")
            loc_label = f"{city_name}, {country}" if country else city_name

            return f"Weather in {loc_label}: {temp}°C, {desc.capitalize()}, Humidity: {humidity}%"
    except httpx.RequestError as e:
        logging.warning("get_weather network error for %s: %s", clean_loc, e)
        return f"Network error occurred while fetching weather for '{clean_loc}'."
    except Exception as e:
        logging.error("get_weather failed for %s: %s", clean_loc, e)
        return f"Could not retrieve weather information for '{clean_loc}'."


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
