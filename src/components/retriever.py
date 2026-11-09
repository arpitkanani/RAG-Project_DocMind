import asyncio
from collections import Counter
import re
import sys
from typing import List, Tuple

import httpx
import yaml
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import AsyncQdrantClient

from src.components.vector_store import VectorStore
from src.exception import (
    CollectionEmbeddingMismatchError,
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

_qdrant_db_cache: dict[str, QdrantVectorStore] = {}


def invalidate_cached_db(collection_name: str):
    _qdrant_db_cache.pop(collection_name, None)


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "give", "tell", "explain", "describe", "define"
}


class Retriever:
    """Handles single-collection and multi-collection retrieval with async search."""

    def __init__(self, collection_names: List[str] | None = None):
        try:
            self.explicit_scope = collection_names is not None
            self.collection_names = [name for name in (collection_names or []) if name]
            self.vs = VectorStore(
                collection_name=self.collection_names[0] if self.collection_names else None
            )
            self.search_type = config["retriever"]["search_type"]
            self.k = config["retriever"]["k"]
            self.fetch_k = config["retriever"].get("fetch_k", self.k)
            self.lambda_mult = config["retriever"].get("lambda_mult", 0.5)
            self.score_threshold = config["retriever"]["score_threshold"]
            self.collection_margin = config["retriever"].get("collection_margin", 0.78)
            self.doc_margin = config["retriever"].get("doc_margin", 0.55)
            self.max_query_variants = config["retriever"].get("max_query_variants", 2)
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query: str) -> List[Document]:
        return asyncio.run(self.aretrieve(query))

    async def aretrieve(self, query: str) -> List[Document]:
        ranked_docs = await self.retrieve_ranked(query)
        return [doc for doc, _, _ in ranked_docs]

    @traceable(run_type="retriever", name="retrieve_ranked")
    async def retrieve_ranked(self, query: str, config: RunnableConfig = None) -> List[Tuple[Document, float, float]]:
        """Retrieve the best chunks with final rank and semantic confidence asynchronously."""
        try:
            logging.info("Async Retrieving for: %s...", query[:50])
            target_collections = await self._resolve_target_collections()
            if not target_collections:
                logging.info("retrieve_ranked: no target collections to search (session has no documents)")
                return []
            query_variants = self._build_query_variants(query)

            tasks = [
                self._search_collection(coll, query, query_variants)
                for coll in target_collections
            ]
            results = await asyncio.gather(*tasks)

            collected_docs: List[Tuple[Document, float]] = []
            for doc_list in results:
                collected_docs.extend(doc_list)

            ranked_docs = self._rerank_documents(query, collected_docs)
            filtered = [
                item for item in ranked_docs
                if item[2] >= self.score_threshold or item[1] >= 12.0
            ]

            if filtered:
                best_by_collection: dict[str, float] = {}
                for doc, final_score, _ in filtered:
                    coll = doc.metadata.get("collection_name")
                    if coll not in best_by_collection or final_score > best_by_collection[coll]:
                        best_by_collection[coll] = final_score

                top_score = max(best_by_collection.values())
                competitive_collections = {
                    coll for coll, score in best_by_collection.items()
                    if score >= top_score * self.collection_margin
                }

                collection_filtered = [
                    item for item in filtered
                    if item[0].metadata.get("collection_name") in competitive_collections
                ]

                docs = []
                for coll in competitive_collections:
                    coll_docs = sorted(
                        [item for item in collection_filtered if item[0].metadata.get("collection_name") == coll],
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    if not coll_docs:
                        continue
                    coll_best = coll_docs[0][1]
                    docs.extend(
                        item for item in coll_docs
                        if item[1] >= coll_best * self.doc_margin or item[2] >= 0.48
                    )

                docs = sorted(docs, key=lambda item: item[1], reverse=True)[: self.k]
            else:
                docs = []

            if not docs and ranked_docs:
                best_doc, best_final_score, best_semantic_score = ranked_docs[0]
                if best_semantic_score >= max(self.score_threshold - 0.05, 0.35) or best_final_score >= 10.0:
                    best_collection = best_doc.metadata.get("collection_name")
                    same_collection_docs = [
                        item for item in ranked_docs
                        if item[0].metadata.get("collection_name") == best_collection
                    ]
                    docs = same_collection_docs[: min(self.k, 3)]

            logging.info(
                "Retrieved %d grounded chunks across %d collections",
                len(docs),
                len(target_collections),
            )
            return docs
        except (CollectionNotFoundError, KnowledgeBaseEmptyError, CollectionEmbeddingMismatchError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    async def _resolve_target_collections(self) -> List[str]:
        # If caller explicitly provided collection_names (even if empty):
        if self.explicit_scope:
            if not self.collection_names:
                # No collections attached to this session -> return empty immediately without global leak
                return []
            available = await self.vs.alist_collections()
            missing = [name for name in self.collection_names if name not in available]
            if missing:
                raise CollectionNotFoundError(missing)
            return self.collection_names

        # Unscoped / global fallback (e.g. CLI tools or background workers)
        available = await self.vs.alist_collections()
        if not available:
            raise KnowledgeBaseEmptyError(
                "No documents found. Please upload a document first."
            )
        return available

    async def _search_collection(
        self,
        collection_name: str,
        query: str,
        query_variants: List[str],
    ) -> List[Tuple[Document, float]]:
        try:
            db = self._get_cached_db(collection_name)
        except CollectionEmbeddingMismatchError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "vector dimension" in err_str or "dimension error" in err_str or "wrong input" in err_str or "dim:" in err_str:
                actual_dim = self.vs.get_collection_vector_dimension(collection_name) or (384 if self.vs.active_specs["dimension"] == 1536 else 1536)
                actual_model = "BAAI/bge-small-en-v1.5" if actual_dim == 384 else "text-embedding-3-small"
                raise CollectionEmbeddingMismatchError(
                    collection_name=collection_name,
                    expected_model=self.vs.active_specs["model"],
                    actual_model=actual_model,
                    expected_dim=self.vs.active_specs["dimension"],
                    actual_dim=actual_dim,
                )
            raise

        async def _search_one_variant(variant: str) -> List[Tuple[Document, float]]:
            v_docs: List[Tuple[Document, float]] = []
            try:
                if self.search_type == "mmr":
                    if self.vs.is_remote:
                        sim_res = await db.asimilarity_search_with_score(variant, k=self.fetch_k)
                        score_lookup = {
                            doc.page_content: self._normalize_similarity_score(score)
                            for doc, score in sim_res
                        }
                        mmr_docs = await db.amax_marginal_relevance_search(
                            variant,
                            k=self.k,
                            fetch_k=self.fetch_k,
                            lambda_mult=self.lambda_mult,
                        )
                    else:
                        sim_res = await asyncio.to_thread(db.similarity_search_with_score, variant, k=self.fetch_k)
                        score_lookup = {
                            doc.page_content: self._normalize_similarity_score(score)
                            for doc, score in sim_res
                        }
                        mmr_docs = await asyncio.to_thread(
                            db.max_marginal_relevance_search,
                            variant,
                            k=self.k,
                            fetch_k=self.fetch_k,
                            lambda_mult=self.lambda_mult,
                        )
                    v_docs.extend(
                        (doc, score_lookup.get(doc.page_content, 0.0)) for doc in mmr_docs
                    )
                else:
                    sim_res = None
                    try:
                        if self.vs.is_remote:
                            sim_res = await db.asimilarity_search_with_score(variant, k=self.fetch_k)
                        else:
                            sim_res = await asyncio.to_thread(db.similarity_search_with_score, variant, k=self.fetch_k)
                    except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                        logging.warning("Qdrant search timeout/network glitch on variant %r: %s. Retrying once...", variant[:30], net_err)
                        await asyncio.sleep(1.0)
                        try:
                            sim_res = await asyncio.to_thread(db.similarity_search_with_score, variant, k=self.fetch_k)
                        except Exception as retry_err:
                            logging.error("Qdrant retry search also failed: %s", retry_err)
                            sim_res = []

                    if sim_res:
                        v_docs.extend(
                            (doc, self._normalize_similarity_score(score))
                            for doc, score in sim_res
                        )
            except CollectionEmbeddingMismatchError:
                raise
            except Exception as search_err:
                err_str = str(search_err).lower()
                if "timeout" in err_str or "timed out" in err_str or "read operation" in err_str:
                    logging.warning("Recovering from Qdrant timeout in collection %s: %s", collection_name, search_err)
                    return []
                if "vector dimension" in err_str or "dimension error" in err_str or "wrong input" in err_str or "dim:" in err_str:
                    actual_dim = self.vs.get_collection_vector_dimension(collection_name) or (384 if self.vs.active_specs["dimension"] == 1536 else 1536)
                    actual_model = "BAAI/bge-small-en-v1.5" if actual_dim == 384 else "text-embedding-3-small"
                    raise CollectionEmbeddingMismatchError(
                        collection_name=collection_name,
                        expected_model=self.vs.active_specs["model"],
                        actual_model=actual_model,
                        expected_dim=self.vs.active_specs["dimension"],
                        actual_dim=actual_dim,
                    )
                logging.warning("Error searching collection %s: %s", collection_name, search_err)
                return []
            return v_docs

        variant_results = await asyncio.gather(*[_search_one_variant(v) for v in query_variants])
        docs: List[Tuple[Document, float]] = []
        for v_res in variant_results:
            docs.extend(v_res)

        for doc, _ in docs:
            doc.metadata.setdefault("collection_name", collection_name)

        # Purge tiny single-word noise chunks (e.g. isolated table cells like 'to', 'chart', 'Quiz:')
        meaningful_docs = [
            (doc, score) for doc, score in docs
            if len(doc.page_content.strip()) >= 25
        ]
        return meaningful_docs or docs

    def _get_cached_db(self, collection_name: str) -> QdrantVectorStore:
        # Enforce dimension compatibility check before returning or constructing vector store
        self.vs.verify_collection_compatibility(collection_name)
        if collection_name not in _qdrant_db_cache:
            client = self.vs._client()
            _qdrant_db_cache[collection_name] = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=self.vs.embedding_model,
            )
        return _qdrant_db_cache[collection_name]

    def _rerank_documents(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float, float]]:
        query_terms = self._tokenize(query)
        meaningful_query_terms = {
            t for t in query_terms if t not in STOP_WORDS and len(t) > 2
        }
        query_phrases = self._extract_phrases(query)
        deduped = {}

        for doc, semantic_score in docs:
            content_clean = doc.page_content.strip()
            # Discard short fragments that cannot provide meaningful context to the LLM
            if len(content_clean) < 25:
                continue

            key = (
                doc.page_content,
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("page", "")),
                str(doc.metadata.get("timestamp", "")),
            )
            content_terms = self._tokenize(doc.page_content)

            # Meaningful keyword overlap calculation
            if meaningful_query_terms:
                matching_terms = meaningful_query_terms & content_terms
                term_overlap = len(matching_terms)
                overlap_ratio = term_overlap / len(meaningful_query_terms)
            else:
                matching_terms = query_terms & content_terms
                term_overlap = len(matching_terms)
                overlap_ratio = term_overlap / max(len(query_terms), 1)

            text_lower = doc.page_content.lower()
            metadata_text = " ".join(
                str(value).lower() for value in doc.metadata.values() if value is not None
            )

            bonus = 0.0
            if "project" in query_terms and "project" in text_lower:
                bonus += 2.0
            if "experience" in query_terms and "experience" in text_lower:
                bonus += 2.0
            if "summary" in query_terms and doc.metadata.get("type") == "youtube":
                bonus += 2.0
            if any(phrase and phrase in text_lower for phrase in query_phrases):
                bonus += 6.0
            if any(term in metadata_text for term in meaningful_query_terms if len(term) > 3):
                bonus += 2.0
            if self._looks_like_structured_answer(query_terms, text_lower):
                bonus += 2.0

            # Exact matching for specific topic keywords: gantt, chart, cost, estimate, etc.
            specific_topics = {"gantt", "chart", "cost", "estimate", "budget", "schedule", "timeline", "milestone", "activities", "duration"}
            topic_matches = query_terms & specific_topics & content_terms
            if topic_matches:
                bonus += min(4.0 * len(topic_matches), 8.0)

            # Specific proper names or non-stopword tokens in query (like "arpit", "dezlor")
            query_entity_tokens = {t for t in meaningful_query_terms if t not in specific_topics and len(t) > 3}
            entity_matches = query_entity_tokens & content_terms
            if entity_matches:
                bonus += min(2.5 * len(entity_matches), 10.0)

            # Author and creator front-matter boosts
            is_author_query = bool(query_terms & {"author", "writer", "written", "creator", "submitted", "prepared", "student", "intern", "who"})
            if is_author_query:
                if any(m in text_lower for m in ["prepared by", "submitted by", "author", "student name", "guided by", "written by", "roll no", "enrollment"]):
                    bonus += 5.0
                if doc.metadata.get("page") in (0, 1) or doc.metadata.get("doc_index", 99) <= 1:
                    bonus += 3.0

            # Project name and title front-matter boost
            is_title_query = bool(query_terms & {"project", "title", "topic", "name"})
            if is_title_query:
                if any(m in text_lower for m in ["project name", "project title", "title of the project", "author"]):
                    bonus += 5.0
                if doc.metadata.get("page") in (0, 1) or doc.metadata.get("doc_index", 99) <= 1:
                    bonus += 3.0

            has_relevance = term_overlap > 0 or semantic_score >= 0.45 or bonus > 0
            locator_bonus = 1.0 if (has_relevance and doc.metadata.get("page") is not None) else 0.0
            heading_bonus = 2.0 if (has_relevance and self._has_heading_signal(text_lower)) else 0.0
            lexical_boost = min(overlap_ratio * 20.0, 20.0)

            # Base score strongly driven by semantic confidence (0.0 to 1.0 -> 0.0 to 60.0)
            base_semantic_score = semantic_score * 60.0
            final_score = base_semantic_score + lexical_boost + bonus + locator_bonus + heading_bonus

            # Index & Glossary page penalty:
            # Prevents textbook index pages or keyword listings (e.g. "Chess, 2 Chord, 575 ... CNN, 326")
            # from dominating explanation chapters when the user did not ask for an index.
            is_index = bool(
                re.search(r"(?i)\bindex\b\s*\n\s*[a-z],\s*\d+", text_lower)
                or re.search(r"(?i)^index\s*\n", text_lower)
                or len(re.findall(r"\b[a-zA-Z\s\-]{3,30},\s*\d{1,4}\b", text_lower)) >= 4
            )
            if is_index and not any(k in query_terms for k in ("index", "page", "where", "location")):
                final_score *= 0.1

            stored = deduped.get(key)
            if stored is None or final_score > stored[0]:
                deduped[key] = (final_score, doc, semantic_score)

        ranked = sorted(
            deduped.values(),
            key=lambda item: item[0],
            reverse=True,
        )
        return [(item[1], item[0], item[2]) for item in ranked]

    @staticmethod
    def _normalize_similarity_score(score: float | None) -> float:
        if score is None:
            return 0.0
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0

        # Cosine similarity in Qdrant ranges in [-1.0, 1.0], typically [0.0, 1.0] for normalized embeddings.
        # Higher score = more similar. Never invert low scores.
        if value < 0.0:
            return 0.0
        if value <= 1.0:
            return value
        return max(0.0, min(1.0, 1.0 / (1.0 + value)))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _extract_phrases(text: str) -> List[str]:
        quoted = re.findall(r'"([^"]+)"', text or "")
        if quoted:
            return [item.strip().lower() for item in quoted if item.strip()]

        words = [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP_WORDS]
        counts = Counter(words)
        phrases = [
            " ".join(words[index : index + 2])
            for index in range(len(words) - 1)
            if counts[words[index]] == 1 and counts[words[index + 1]] == 1
        ]
        return phrases[:3]

    def _build_query_variants(self, query: str) -> List[str]:
        normalized = " ".join((query or "").split())
        tokens = re.findall(r"[a-z0-9]+", normalized.lower())
        variants = [normalized]

        significant = [token for token in tokens if token not in STOP_WORDS and len(token) > 2]
        if significant:
            variants.append(" ".join(significant[:8]))
        if len(tokens) > 5:
            variants.append(" ".join(tokens[:5]))

        # Domain-aware query expansions
        token_set = set(tokens)
        ACRONYM_MAP = {
            "cnn": "convolutional neural network convolutional networks",
            "cnns": "convolutional neural networks convolutional networks",
            "rnn": "recurrent neural network",
            "rnns": "recurrent neural networks",
            "lstm": "long short term memory",
            "gan": "generative adversarial network",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "llm": "large language model",
        }
        for token in tokens:
            if token in ACRONYM_MAP:
                variants.append(ACRONYM_MAP[token])
                variants.append(f"{token} {ACRONYM_MAP[token]}")

        if token_set & {"author", "writer", "written", "creator", "submitted", "prepared", "student", "who"}:
            variants.append("submitted by prepared by author student name")
            variants.append("prepared by")
            variants.append("submitted by")
        if token_set & {"project", "title", "name", "topic"}:
            variants.append("project title overview abstract report")
            variants.append("project name author")
        if token_set & {"gantt", "chart", "schedule", "timeline"}:
            variants.append("gantt chart project schedule timeline activities weeks")
        if token_set & {"cost", "estimate", "budget", "price"}:
            variants.append("cost estimate project budget expenditure total cost")

        # Proper names / distinct entities in query
        proper_names = [w.lower() for w in re.findall(r"\b[A-Z][a-z0-9]+\b", query) if w.lower() not in STOP_WORDS]
        if proper_names:
            variants.append(f"{' '.join(proper_names)} project name author")

        deduped = [
            variant for index, variant in enumerate(variants)
            if variant and variant not in variants[:index]
        ]
        return deduped[: self.max_query_variants]

    @staticmethod
    def _looks_like_structured_answer(query_terms: set[str], text_lower: str) -> bool:
        structured_terms = {
            "list", "steps", "points", "summary", "table",
            "responsibilities", "skills", "experience", "projects", "education",
        }
        return bool(query_terms & structured_terms) and any(
            marker in text_lower
            for marker in [":", "-", "1.", "2.", "project", "experience", "summary"]
        )

    @staticmethod
    def _has_heading_signal(text_lower: str) -> bool:
        lines = [line.strip() for line in text_lower.splitlines() if line.strip()]
        if not lines:
            return False
        first_line = lines[0]
        return len(first_line) < 80 and any(char.isalpha() for char in first_line)

    @traceable(run_type="retriever", name="get_full_context")
    async def get_full_context(self, max_chars: int = 6000) -> List[Document]:
        """Return chunks in original document order asynchronously."""
        try:
            target_collections = await self._resolve_target_collections()
            if not target_collections:
                logging.info("get_full_context: no target collections to retrieve (session has no documents)")
                return []

            all_docs: List[Document] = []
            for collection_name in target_collections:
                vs = VectorStore(collection_name=collection_name)
                docs = await vs.aget_all_documents()
                docs.sort(key=lambda d: d.metadata.get("doc_index", 0))
                all_docs.extend(docs)

            total_chars = 0
            selected: List[Document] = []
            for doc in all_docs:
                doc_len = len(doc.page_content)
                if total_chars + doc_len > max_chars and selected:
                    break
                selected.append(doc)
                total_chars += doc_len

            return selected
        except (CollectionNotFoundError, KnowledgeBaseEmptyError, CollectionEmbeddingMismatchError):
            raise
        except Exception as e:
            raise CustomException(e, sys)
