import logging
from typing import List, Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-Encoder Reranker for fine-grained query-document semantic relevance scoring."""

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    def _get_model(self):
        if not self._load_attempted:
            self._load_attempted = True
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from sentence_transformers import CrossEncoder
                    logger.info(f"Loading CrossEncoder reranker model: {self.model_name}")
                    self._model = CrossEncoder(self.model_name, max_length=512)
            except BaseException as e:
                logger.info(f"CrossEncoder not active ({type(e).__name__}). Using hybrid RRF ranker.")
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = settings.RERANKER_TOP_K
    ) -> List[Dict[str, Any]]:
        """Reranks a list of retrieved documents based on cross-encoder relevance scores."""
        if not documents:
            return []

        model = self._get_model()
        if model is None:
            # Fallback: preserve original rank and return top_k
            return documents[:top_k]

        try:
            # Prepare pairs
            pairs = [[query, doc["text"]] for doc in documents]
            scores = model.predict(pairs)

            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = round(float(score), 4)
                scored_docs.append(doc_copy)

            # Sort by rerank score descending
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_docs[:top_k]

        except Exception as e:
            logger.warning(f"Error during reranking ({e}). Falling back to original retrieval order.")
            return documents[:top_k]
