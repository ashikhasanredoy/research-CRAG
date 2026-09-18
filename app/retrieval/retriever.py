import logging
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.config import settings
from app.vectorstore.chroma import ChromaVectorStoreManager

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid Retriever combining Dense ChromaDB Vector Search and Sparse BM25 Keyword Search."""

    def __init__(self, vectorstore: Optional[ChromaVectorStoreManager] = None):
        self.vectorstore = vectorstore or ChromaVectorStoreManager()
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict[str, Any]] = []
        self.refresh_bm25_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())

    def refresh_bm25_index(self) -> None:
        """Rebuilds BM25 index from current ChromaDB documents."""
        all_docs = self.vectorstore.get_all_documents()
        self._bm25_docs = all_docs
        if all_docs:
            tokenized_corpus = [self._tokenize(doc["text"]) for doc in all_docs]
            self._bm25_index = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25 index with {len(all_docs)} documents.")
        else:
            self._bm25_index = None

    def retrieve_dense(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """Performs dense vector retrieval."""
        return self.vectorstore.similarity_search_with_score(query=query, k=top_k)

    def retrieve_sparse(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """Performs sparse BM25 retrieval."""
        if not self._bm25_index or not self._bm25_docs:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25_index.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        sparse_results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                doc = self._bm25_docs[idx]
                sparse_results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": round(float(scores[idx]), 4),
                    "source_type": "bm25"
                })
        return sparse_results

    def retrieve(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """Performs Hybrid Search using Reciprocal Rank Fusion (RRF) between Dense & Sparse."""
        # Ensure BM25 matches ChromaDB count
        if self._bm25_docs is None or len(self._bm25_docs) != self.vectorstore.count():
            self.refresh_bm25_index()

        dense_docs = self.retrieve_dense(query, top_k=top_k * 2)
        sparse_docs = self.retrieve_sparse(query, top_k=top_k * 2)

        # If only one source has documents, return it
        if not sparse_docs:
            return dense_docs[:top_k]
        if not dense_docs:
            return sparse_docs[:top_k]

        # Reciprocal Rank Fusion
        rrf_constant = 60
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_docs):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight * (1.0 / (rrf_constant + rank + 1))

        for rank, doc in enumerate(sparse_docs):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight * (1.0 / (rrf_constant + rank + 1))

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda d_id: scores[d_id], reverse=True)

        fused_docs = []
        for d_id in sorted_ids[:top_k]:
            item = dict(doc_map[d_id])
            item["hybrid_score"] = round(scores[d_id], 6)
            fused_docs.append(item)

        return fused_docs
