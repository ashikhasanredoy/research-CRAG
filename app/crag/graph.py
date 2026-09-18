import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from app.config import settings
from app.crag.state import CRAGState
from app.crag.nodes import CRAGNodes

logger = logging.getLogger(__name__)

class CRAGWorkflow:
    """Builds and runs the Corrective RAG (CRAG) workflow matching the exact architecture."""

    def __init__(self, nodes: Optional[CRAGNodes] = None):
        self.nodes = nodes or CRAGNodes()
        self.app = self._build_graph()

    def _route_after_confidence_check(self, state: CRAGState) -> str:
        """Routes to Fast Path (Skip Grader) or to Retrieval Grader LLM."""
        if state.get("confidence_level") == "HIGH" or state.get("fast_path_used"):
            logger.info("Confidence Estimator: HIGH confidence. Fast-path routing to aggregate_context.")
            return "aggregate_context"
        logger.info("Confidence Estimator: LOW confidence. Routing to grade_documents (LLM Call #1).")
        return "grade_documents"

    def _route_after_grader(self, state: CRAGState) -> str:
        """Routes to Accept or to Query Rewriter."""
        status = state.get("retrieval_status", "bad")
        if status == "good":
            logger.info("Retrieval Grader: RELEVANT docs found. Routing to aggregate_context.")
            return "aggregate_context"
        logger.info("Retrieval Grader: IRRELEVANT docs. Routing to rewrite_query (LLM Call #2).")
        return "rewrite_query"

    def _route_after_re_retrieval(self, state: CRAGState) -> str:
        """Routes to Accept (FOUND) or Web Search Fallback (NOT FOUND)."""
        status = state.get("retrieval_status", "bad")
        if status == "good":
            logger.info("Re-Retrieval: FOUND relevant documents. Routing to aggregate_context.")
            return "aggregate_context"
        
        if settings.ENABLE_WEB_SEARCH_FALLBACK:
            logger.info("Re-Retrieval: NOT FOUND. Routing to web_search_node.")
            return "web_search_node"
        
        return "aggregate_context"

    def _build_graph(self):
        """Assembles StateGraph nodes and edges matching the architecture diagram."""
        workflow = StateGraph(CRAGState)

        # 1. Add All Nodes
        workflow.add_node("preprocess_query", self.nodes.preprocess_query)
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents)
        workflow.add_node("estimate_confidence", self.nodes.estimate_confidence)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("rewrite_query", self.nodes.rewrite_query)
        workflow.add_node("re_retrieve_and_check", self.nodes.re_retrieve_and_check)
        workflow.add_node("web_search_node", self.nodes.web_search_node)
        workflow.add_node("aggregate_context", self.nodes.aggregate_context)
        workflow.add_node("generate_answer", self.nodes.generate_answer)
        workflow.add_node("verify_citations", self.nodes.verify_citations)

        # 2. Set Entry Point
        workflow.set_entry_point("preprocess_query")

        # 3. Define Flow
        workflow.add_edge("preprocess_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "estimate_confidence")

        # Routing from Confidence Estimator
        workflow.add_conditional_edges(
            "estimate_confidence",
            self._route_after_confidence_check,
            {
                "aggregate_context": "aggregate_context",
                "grade_documents": "grade_documents"
            }
        )

        # Routing from Grader (LLM Call #1)
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grader,
            {
                "aggregate_context": "aggregate_context",
                "rewrite_query": "rewrite_query"
            }
        )

        # Routing from Rewriter (LLM Call #2) -> Re-Retrieval Check
        workflow.add_edge("rewrite_query", "re_retrieve_and_check")

        workflow.add_conditional_edges(
            "re_retrieve_and_check",
            self._route_after_re_retrieval,
            {
                "aggregate_context": "aggregate_context",
                "web_search_node": "web_search_node"
            }
        )

        # Web Search -> Context Aggregator
        workflow.add_edge("web_search_node", "aggregate_context")

        # Context Aggregator -> Answer Generator (LLM Call #3) -> Verification -> END
        workflow.add_edge("aggregate_context", "generate_answer")
        workflow.add_edge("generate_answer", "verify_citations")
        workflow.add_edge("verify_citations", END)

        return workflow.compile()

    def run(self, question: str) -> Dict[str, Any]:
        """Executes the complete CRAG workflow on a user query."""
        initial_state: CRAGState = {
            "question": question,
            "original_question": question,
            "cleaned_query": question,
            "rewritten_query": None,
            "rewrite_count": 0,
            "raw_documents": [],
            "retrieval_confidence": 0.0,
            "confidence_level": "LOW",
            "fast_path_used": False,
            "graded_documents": [],
            "irrelevant_documents": [],
            "retrieval_status": "pending",
            "context": "",
            "generation": "",
            "verification_result": {},
            "citations": [],
            "web_search_used": False,
            "trace": []
        }

        final_state = self.app.invoke(initial_state)
        return final_state
