import re
import datetime
import logging
from typing import Dict, Any, List
from app.config import settings
from app.crag.state import CRAGState
from app.retrieval.retriever import HybridRetriever
from app.crag.grader import RetrievalGrader
from app.crag.query_rewriter import QueryRewriter
from app.crag.web_search import WebSearchFallback
from app.crag.generator import GroundedAnswerGenerator
from app.crag.verifier import AnswerVerifier

logger = logging.getLogger(__name__)

class CRAGNodes:
    """Implements all workflow nodes for the Corrective RAG state machine."""

    def __init__(
        self,
        retriever: HybridRetriever = None,
        grader: RetrievalGrader = None,
        query_rewriter: QueryRewriter = None,
        web_search: WebSearchFallback = None,
        generator: GroundedAnswerGenerator = None,
        verifier: AnswerVerifier = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.grader = grader or RetrievalGrader()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.web_search = web_search or WebSearchFallback()
        self.generator = generator or GroundedAnswerGenerator()
        self.verifier = verifier or AnswerVerifier()

    def _add_trace(self, state: CRAGState, step_name: str, details: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace = list(state.get("trace", []))
        trace.append({
            "step": step_name,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "details": details
        })
        return trace

    # 1. Query Preprocessing
    def preprocess_query(self, state: CRAGState) -> Dict[str, Any]:
        """Cleans, normalizes whitespace and symbols in the user query."""
        raw_q = state.get("question", "").strip()
        cleaned = re.sub(r"\s+", " ", raw_q)
        original = state.get("original_question") or cleaned

        trace = self._add_trace(state, "query_preprocessing", {
            "raw_query": raw_q,
            "cleaned_query": cleaned
        })

        return {
            "question": cleaned,
            "cleaned_query": cleaned,
            "original_question": original,
            "rewrite_count": state.get("rewrite_count", 0),
            "trace": trace
        }

    # 2. Hybrid Retrieval (Vector + BM25 -> RRF Fusion)
    def retrieve_documents(self, state: CRAGState) -> Dict[str, Any]:
        """Performs hybrid vector + BM25 retrieval with Reciprocal Rank Fusion."""
        query = state.get("question")
        raw_docs = self.retriever.retrieve(query=query, top_k=settings.RETRIEVAL_TOP_K)

        top_scores = [d.get("hybrid_score") or d.get("score") for d in raw_docs[:3]]
        trace = self._add_trace(state, "hybrid_retrieval", {
            "query": query,
            "documents_retrieved": len(raw_docs),
            "top_rrf_scores": top_scores
        })

        return {
            "raw_documents": raw_docs,
            "trace": trace
        }

    # 3. Retrieval Confidence Estimator
    def estimate_confidence(self, state: CRAGState) -> Dict[str, Any]:
        """Estimates retrieval confidence to decide Fast Path vs LLM Grader."""
        docs = state.get("raw_documents", [])
        if not docs:
            trace = self._add_trace(state, "confidence_estimator", {
                "confidence": 0.0,
                "level": "LOW",
                "decision": "Route to Grader / Rewriter"
            })
            return {
                "retrieval_confidence": 0.0,
                "confidence_level": "LOW",
                "fast_path_used": False,
                "retrieval_status": "bad",
                "trace": trace
            }

        top_score = docs[0].get("hybrid_score", 0.0) or docs[0].get("score", 0.0)
        q_words = set(state["question"].lower().split())
        top_text_words = set(docs[0].get("text", "").lower().split())
        overlap_ratio = len(q_words.intersection(top_text_words)) / max(len(q_words), 1)

        # High confidence if RRF score is strong or keyword overlap is high
        is_high_confidence = top_score >= settings.CONFIDENCE_THRESHOLD or overlap_ratio >= 0.40

        conf_level = "HIGH" if is_high_confidence else "LOW"
        fast_path = is_high_confidence

        trace = self._add_trace(state, "confidence_estimator", {
            "top_rrf_score": top_score,
            "keyword_overlap": round(overlap_ratio, 2),
            "confidence_level": conf_level,
            "fast_path_used": fast_path
        })

        graded_docs = docs[:settings.RERANKER_TOP_K] if fast_path else []

        return {
            "retrieval_confidence": round(top_score, 4),
            "confidence_level": conf_level,
            "fast_path_used": fast_path,
            "graded_documents": graded_docs,
            "retrieval_status": "good" if fast_path else "pending",
            "trace": trace
        }

    # 4. Retrieval Grader (LLM Call #1)
    def grade_documents(self, state: CRAGState) -> Dict[str, Any]:
        """Evaluates document relevance using fast batch LLM grading."""
        query = state.get("original_question") or state["question"]
        raw_docs = state.get("raw_documents", [])

        relevant_docs, irrelevant_docs, status = self.grader.grade_documents(
            question=query,
            documents=raw_docs[:settings.RERANKER_TOP_K]
        )

        trace = self._add_trace(state, "retrieval_grader_llm", {
            "relevant_count": len(relevant_docs),
            "irrelevant_count": len(irrelevant_docs),
            "verdict": status
        })

        return {
            "graded_documents": relevant_docs,
            "irrelevant_documents": irrelevant_docs,
            "retrieval_status": status,
            "trace": trace
        }

    # 5. Query Rewriter (LLM Call #2)
    def rewrite_query(self, state: CRAGState) -> Dict[str, Any]:
        """Reformulates query into domain-specific academic terminology."""
        orig_q = state.get("original_question") or state["question"]
        current_rewrites = state.get("rewrite_count", 0) + 1

        new_query = self.query_rewriter.rewrite(orig_q)

        trace = self._add_trace(state, "query_rewriter_llm", {
            "original_query": orig_q,
            "rewritten_query": new_query,
            "rewrite_attempt": current_rewrites
        })

        return {
            "question": new_query,
            "rewritten_query": new_query,
            "rewrite_count": current_rewrites,
            "trace": trace
        }

    # 6. Re-Retrieval & Confidence Check
    def re_retrieve_and_check(self, state: CRAGState) -> Dict[str, Any]:
        """Re-retrieves documents using rewritten query and evaluates confidence."""
        rewritten_q = state.get("question")
        re_docs = self.retriever.retrieve(query=rewritten_q, top_k=settings.RETRIEVAL_TOP_K)

        has_found = False
        if re_docs:
            top_score = re_docs[0].get("hybrid_score", 0.0) or re_docs[0].get("score", 0.0)
            has_found = top_score >= 0.010 or len(re_docs) > 0

        trace = self._add_trace(state, "re_retrieval_check", {
            "query": rewritten_q,
            "docs_found": len(re_docs),
            "status": "FOUND" if has_found else "NOT FOUND"
        })

        if has_found:
            return {
                "raw_documents": re_docs,
                "graded_documents": re_docs[:settings.RERANKER_TOP_K],
                "retrieval_status": "good",
                "trace": trace
            }
        else:
            return {
                "raw_documents": re_docs,
                "retrieval_status": "bad",
                "trace": trace
            }

    # 7. Web Search Fallback (External Context)
    def web_search_node(self, state: CRAGState) -> Dict[str, Any]:
        """Fetches external academic context via DuckDuckGo when local DB lacks info."""
        orig_q = state.get("original_question") or state["question"]
        web_docs = self.web_search.search(query=orig_q)

        trace = self._add_trace(state, "web_search_fallback", {
            "query": orig_q,
            "web_docs_count": len(web_docs)
        })

        return {
            "graded_documents": web_docs,
            "web_search_used": True,
            "retrieval_status": "web_fallback" if web_docs else "bad",
            "trace": trace
        }

    # 8. Context Aggregator (Local + Web)
    def aggregate_context(self, state: CRAGState) -> Dict[str, Any]:
        """Aggregates verified local and web documents into formatted context with citations."""
        docs = state.get("graded_documents", [])
        context_str, citations = self.generator.build_context(docs)

        trace = self._add_trace(state, "context_aggregator", {
            "total_sources": len(docs),
            "citations_extracted": len(citations)
        })

        return {
            "context": context_str,
            "citations": citations,
            "trace": trace
        }

    # 9. Answer Generator (LLM Call #3 - Llama 3.2)
    def generate_answer(self, state: CRAGState) -> Dict[str, Any]:
        """Synthesizes grounded answer using Llama 3.2."""
        query = state.get("original_question") or state["question"]
        docs = state.get("graded_documents", [])

        answer, citations, context_str = self.generator.generate(
            question=query,
            documents=docs
        )

        trace = self._add_trace(state, "answer_generator_llm", {
            "answer_length": len(answer),
            "citations_count": len(citations)
        })

        return {
            "generation": answer,
            "citations": citations,
            "context": context_str,
            "trace": trace
        }

    # 10. Citation & Source Verification
    def verify_citations(self, state: CRAGState) -> Dict[str, Any]:
        """Verifies groundedness and in-text page citations."""
        query = state.get("original_question") or state["question"]
        answer = state.get("generation", "")
        context = state.get("context", "")

        verification = self.verifier.verify(
            question=query,
            answer=answer,
            context=context
        )

        trace = self._add_trace(state, "citation_verification", {
            "faithfulness_score": verification.get("faithfulness_score", 1.0),
            "reasoning": verification.get("reasoning", "")
        })

        return {
            "verification_result": verification,
            "trace": trace
        }
