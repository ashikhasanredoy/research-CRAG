from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class TraceStep(TypedDict):
    step_name: str
    timestamp: str
    details: Dict[str, Any]

class CRAGState(TypedDict):
    # User Query & Preprocessing
    question: str
    original_question: str
    cleaned_query: str
    rewritten_query: Optional[str]
    rewrite_count: int

    # Hybrid Retrieval & Top-K
    raw_documents: List[Dict[str, Any]]
    retrieval_confidence: float
    confidence_level: str  # "HIGH", "LOW"
    fast_path_used: bool

    # Evaluation & External Fallback
    graded_documents: List[Dict[str, Any]]
    irrelevant_documents: List[Dict[str, Any]]
    retrieval_status: str  # "good", "bad", "web_fallback"
    web_search_used: bool

    # Context Aggregator & Generation
    context: str
    generation: str
    verification_result: Dict[str, Any]
    citations: List[Dict[str, Any]]

    # Execution Trace
    trace: List[Dict[str, Any]]
