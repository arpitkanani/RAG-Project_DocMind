import re
import sys
from collections import Counter
import threading
from typing import List, Tuple

import yaml
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from src.components.vector_store import VectorStore
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Process-wide cache: QdrantVectorStore.from_existing_collection() secretly
# costs one embedding API call every time it's constructed (LangChain embeds
# a "dummy_text" string to validate vector dimensions match). Caching by
# collection name means that validation cost is paid ONCE per collection
# per process lifetime, not once per query -- this was the single biggest
# source of wasted embedding calls against the shared rate limit.
_qdrant_db_cache: dict[str, QdrantVectorStore] = {}


def invalidate_cached_db(collection_name: str):
    """Call this whenever a collection is deleted, so a future re-upload
    with the same name doesn't serve a stale cached connection."""
    _qdrant_db_cache.pop(collection_name, None)

from sentence_transformers import CrossEncoder

_reranker = None
_reranker_lock = threading.Lock()

def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    return _reranker

class Retriever:
    """Handles single-collection and multi-collection retrieval."""

    def __init__(self, collection_names: List[str] | None = None):
        try:
            logging.info("Initializing Retriever")
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
            logging.info("Retriever initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve the best chunks across one or more collections."""
        try:
            ranked_docs = self.retrieve_ranked(query)
            return [doc for doc, _, _ in ranked_docs]
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve_ranked(self, query: str) -> List[Tuple[Document, float, float]]:
        """Retrieve the best chunks with final rank and semantic confidence."""
        try:
            logging.info("Retrieving for: %s...", query[:50])
            target_collections = self._resolve_target_collections()
            query_variants = self._build_query_variants(query)

            collected_docs: List[Tuple[Document, float]] = []
            for collection_name in target_collections:
                collected_docs.extend(
                    self._search_collection(collection_name, query, query_variants)
                )

            ranked_docs = self._rerank_documents(query, collected_docs)
            
            filtered = [item for item in ranked_docs if item[2] >= self.score_threshold]

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

                # Per-document margin — even within the same (competitive) collection,
                # only keep chunks close to that collection's OWN best match. Stops
                # weak neighboring pages from riding along with the one strong page.
                DOC_MARGIN = self.doc_margin
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
                        item for item in coll_docs if item[1] >= coll_best * DOC_MARGIN
                    )

                docs = sorted(docs, key=lambda item: item[1], reverse=True)[: self.k]
            else:
                docs = []

            if not docs and ranked_docs:
                best_doc, best_final_score, best_semantic_score = ranked_docs[0]
                logging.info(
                    "No chunks met threshold %.2f. Best semantic score: %.4f | final score: %.4f",
                    self.score_threshold,
                    best_semantic_score,
                    best_final_score,
                )

                if best_semantic_score >= max(self.score_threshold - 0.05, 0.4):
                    best_collection = best_doc.metadata.get("collection_name")
                    same_collection_docs = [
                        item for item in ranked_docs
                        if item[0].metadata.get("collection_name") == best_collection
                    ]
                    docs = same_collection_docs[: min(self.k, 3)]
                    logging.info(
                        "Using top %d chunks from collection '%s' through adaptive fallback",
                        len(docs),
                        best_collection,
                    )

            logging.info(
                "Retrieved %d grounded chunks across %d collections",
                len(docs),
                len(target_collections),
            )
            return docs
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve chunks with similarity scores from the primary collection."""
        try:
            logging.info("Retrieving with scores: %s...", query[:50])
            db = self.vs.get_vectordb()
            results = db.similarity_search_with_score(query, k=self.k)
            logging.info("Retrieved %d chunks with scores", len(results))
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def _resolve_target_collections(self) -> List[str]:
        client = QdrantClient(url=self.vs.qdrant_url)
        try:
            available = [c.name for c in client.get_collections().collections]
        finally:
            client.close()

        if not available:
            raise KnowledgeBaseEmptyError(
                "No documents found. Please upload a document first."
            )

        if self.collection_names:
            missing = [name for name in self.collection_names if name not in available]
            if missing:
                raise CollectionNotFoundError(missing)
            return self.collection_names

        return available

    def _search_collection(
        self,
        collection_name: str,
        query: str,
        query_variants: List[str],
    ) -> List[Tuple[Document, float]]:
        db = self._get_cached_db(collection_name)

        docs: List[Tuple[Document, float]] = []
        for variant in query_variants:
            if self.search_type == "mmr":
                # max_marginal_relevance_search() picks a diverse subset but
                # doesn't return similarity scores -- and retrieve_ranked()
                # later filters everything below score_threshold using that
                # score. A hardcoded 0.0 here meant every MMR result always
                # failed that filter, so MMR silently returned nothing.
                # Fix: look up each selected chunk's real score from a
                # similarity search over the same fetch_k candidate pool
                # MMR chose from (safe to key by exact chunk text, since
                # chunks are unique within one collection).
                score_lookup = {
                    doc.page_content: self._normalize_similarity_score(score)
                    for doc, score in db.similarity_search_with_score(variant, k=self.fetch_k)
                }
                mmr_docs = db.max_marginal_relevance_search(
                    variant,
                    k=self.k,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult,
                )
                docs.extend(
                    (doc, score_lookup.get(doc.page_content, 0.0)) for doc in mmr_docs
                )
            else:
                docs.extend(self._similarity_search_with_scores(db, variant))

        for doc, _ in docs:
            doc.metadata.setdefault("collection_name", collection_name)

        return docs

    def _get_cached_db(self, collection_name: str) -> QdrantVectorStore:
        """Reuse an existing connection if we've already validated this
        collection this process lifetime -- avoids paying the hidden
        dummy-text embedding cost on every single query."""
        if collection_name not in _qdrant_db_cache:
            _qdrant_db_cache[collection_name] = QdrantVectorStore.from_existing_collection(
                embedding=self.vs.embedding_model,
                collection_name=collection_name,
                url=self.vs.qdrant_url,
            )
        return _qdrant_db_cache[collection_name]

    def _rerank_documents(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float, float]]:
        query_terms = self._tokenize(query)
        query_phrases = self._extract_phrases(query)
        deduped = {}

        for doc, semantic_score in docs:
            key = (
                doc.page_content,
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("page", "")),
                str(doc.metadata.get("timestamp", "")),
            )
            content_terms = self._tokenize(doc.page_content)
            term_overlap = len(query_terms & content_terms)
            overlap_ratio = term_overlap / max(len(query_terms), 1)

            text_lower = doc.page_content.lower()
            metadata_text = " ".join(
                str(value).lower() for value in doc.metadata.values() if value is not None
            )

            bonus = 0
            if "project" in query_terms and "project" in text_lower:
                bonus += 4
            if "experience" in query_terms and "experience" in text_lower:
                bonus += 3
            if "summary" in query_terms and doc.metadata.get("type") == "youtube":
                bonus += 2
            if any(phrase and phrase in text_lower for phrase in query_phrases):
                bonus += 8
            if any(term in metadata_text for term in query_terms if len(term) > 3):
                bonus += 3
            if self._looks_like_structured_answer(query_terms, text_lower):
                bonus += 4

            locator_bonus = 1 if doc.metadata.get("page") is not None else 0
            heading_bonus = 2 if self._has_heading_signal(text_lower) else 0
            density_bonus = min(int(overlap_ratio * 10), 6)
            lexical_score = (
                term_overlap * 5 + bonus + locator_bonus + heading_bonus + density_bonus
            )
            final_score = lexical_score + (semantic_score * 12)

            stored = deduped.get(key)
            if stored is None or final_score > stored[0]:
                deduped[key] = (final_score, doc, semantic_score)

        ranked = sorted(
            deduped.values(),
            key=lambda item: item[0],
            reverse=True,
        )
        return [(item[1], item[0], item[2]) for item in ranked]

    def _similarity_search_with_scores(
        self,
        db: QdrantVectorStore,
        query: str,
    ) -> List[Tuple[Document, float]]:
        try:
            results = db.similarity_search_with_score(query, k=self.fetch_k)
            return [
                (doc, self._normalize_similarity_score(score))
                for doc, score in results
            ]
        except Exception:
            logging.exception("Similarity search with scores failed")
            raise

    @staticmethod
    def _normalize_similarity_score(score: float | None) -> float:
        if score is None:
            return 0.0
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        if value < 0:
            return 0.0
        if value <= 1:
            return value
        return max(0.0, min(1.0, 1 / (1 + value)))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _extract_phrases(text: str) -> List[str]:
        quoted = re.findall(r'"([^"]+)"', text or "")
        if quoted:
            return [item.strip().lower() for item in quoted if item.strip()]

        words = re.findall(r"[a-z0-9]+", (text or "").lower())
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

        significant = [token for token in tokens if len(token) > 3]
        if significant:
            variants.append(" ".join(significant[:8]))

        if len(tokens) > 5:
            variants.append(" ".join(tokens[:5]))

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

    def get_full_context(self, max_chars: int = 6000) -> List[Document]:
        """Return chunks in original document order, not similarity-ranked."""
        try:
            target_collections = self._resolve_target_collections()

            all_docs: List[Document] = []
            for collection_name in target_collections:
                vs = VectorStore(collection_name=collection_name)
                docs = vs.get_all_documents()
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

            logging.info(
                "Full-context retrieval: %d chunks (%d chars) across %d collections",
                len(selected),
                total_chars,
                len(target_collections),
            )
            return selected
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)