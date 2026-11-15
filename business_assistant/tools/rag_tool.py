from typing import Optional
from langchain_core.tools import tool

from src.components.retriever import Retriever
from src.exception import CollectionNotFoundError, KnowledgeBaseEmptyError
from src.logger import logging


def get_brochure_rag_tool(bot_id: str):
    """
    Creates an LLM-callable RAG tool bound to the specific deployed bot's
    brochures vector collection (biz_{bot_id}).
    Reuses existing Retriever from src.components.retriever without modifying core RAG code.
    """
    collection_name = f"biz_{bot_id}"

    @tool
    async def search_business_brochures(query: str) -> str:
        """Search the business brochures, product manuals, and catalogs for verified facts, pricing, usage instructions, and benefits.

        Args:
            query: The specific question or topic to search for in the brochures.
        """
        logging.info("[Brochure RAG Tool] Query: %r | Collection: %s", query, collection_name)
        try:
            retriever = Retriever(collection_names=[collection_name])
            ranked_docs = await retriever.retrieve_ranked(query)
            if not ranked_docs:
                return "No matching information found in the uploaded brochures."

            excerpts = []
            for doc, score, col in ranked_docs[:5]:
                content = doc.page_content.strip()
                page = doc.metadata.get("page", "unknown")
                source = doc.metadata.get("source", "brochure.pdf")
                excerpts.append(f"[{source}, Page {page}]:\n{content}")

            return "\n\n---\n\n".join(excerpts)

        except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
            logging.warning("[Brochure RAG Tool] Collection not found or empty: %s", e)
            return "No brochures available for this business."
        except Exception as e:
            logging.exception("[Brochure RAG Tool] Unexpected retrieval error")
            return f"Error searching brochures: {str(e)}"

    return search_business_brochures
