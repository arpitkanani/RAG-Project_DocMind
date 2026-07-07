import re
import sys
from collections import Counter
from typing import List, Tuple

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb as chroma_client

from src.components.vector_store import VectorStore
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


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
            docs = [
                item for item in ranked_docs if item[2] >= self.score_threshold
            ][: self.k]

            if not docs and ranked_docs:
                best_doc, best_final_score, best_semantic_score = ranked_docs[0]
                logging.info(
                    "No chunks met threshold %.2f. Best semantic score: %.4f | final score: %.4f",
                    self.score_threshold,
                    best_semantic_score,
                    best_final_score,
                )

                if best_semantic_score >= max(self.score_threshold - 0.15, 0.2):
                    docs = ranked_docs[: min(self.k, 3)]
                    logging.info(
                        "Using top %d chunks through adaptive fallback",
                        len(docs),
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
        

        client = chroma_client.PersistentClient(path=self.vs.persist_dir)
        available = [collection.name for collection in client.list_collections()]

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
        db = Chroma(
            persist_directory=self.vs.persist_dir,
            embedding_function=self.vs.embedding_model,
            collection_name=collection_name,
        )

        docs: List[Tuple[Document, float]] = []
        for variant in query_variants:
            if self.search_type == "mmr":
                mmr_docs = db.max_marginal_relevance_search(
                        variant,
                        k=self.k,
                        fetch_k=self.fetch_k,
                        lambda_mult=self.lambda_mult,
                    )
                docs.extend((doc, 0.0) for doc in mmr_docs)
            else:
                docs.extend(self._similarity_search_with_scores(db, variant))

        for doc, _ in docs:
            doc.metadata.setdefault("collection_name", collection_name)

        return docs

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
        db: Chroma,
        query: str,
    ) -> List[Tuple[Document, float]]:
        try:
            if hasattr(db, "similarity_search_with_relevance_scores"):
                results = db.similarity_search_with_relevance_scores(
                    query,
                    k=self.fetch_k,
                )
                return [
                    (doc, self._normalize_similarity_score(score))
                    for doc, score in results
                ]

            results = db.similarity_search_with_score(query, k=self.fetch_k)
            return [
                (doc, self._distance_to_similarity(score))
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

    @classmethod
    def _distance_to_similarity(cls, score: float | None) -> float:
        return cls._normalize_similarity_score(score)

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

    @staticmethod
    def _build_query_variants(query: str) -> List[str]:
        normalized = " ".join((query or "").split())
        tokens = re.findall(r"[a-z0-9]+", normalized.lower())
        variants = [normalized]

        significant = [token for token in tokens if len(token) > 3]
        if significant:
            variants.append(" ".join(significant[:8]))

        if len(tokens) > 5:
            variants.append(" ".join(tokens[:5]))

        return [variant for index, variant in enumerate(variants) if variant and variant not in variants[:index]]

    @staticmethod
    def _looks_like_structured_answer(query_terms: set[str], text_lower: str) -> bool:
        structured_terms = {
            "list",
            "steps",
            "points",
            "summary",
            "table",
            "responsibilities",
            "skills",
            "experience",
            "projects",
            "education",
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
