import logging
from app.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Reformulates conversational or underspecified queries into high-recall academic search queries."""

    REWRITE_PROMPT = """You are an AI research assistant optimizing queries for academic paper retrieval in computer vision and machine learning.

Original Question:
\"{question}\"

Reason for rewrite: The previous vector retrieval did not find sufficiently relevant research paper sections.

Task:
Rewrite the question into a clear, keyword-dense, academic search query suitable for semantic vector and keyword search over AI/ML research papers.
Include specific technical terms, model names (e.g., YOLO, CNN, Transformer), methods, or domain keywords (e.g., adverse weather, dehazing, low-light detection, Dark Channel Prior).

Provide ONLY the rewritten query text, with NO preamble, NO quotes, and NO commentary.
"""

    def __init__(self, llm_client: OllamaClient = None):
        self.llm = llm_client or OllamaClient()

    def rewrite(self, question: str) -> str:
        """Generates an improved academic query string."""
        prompt = self.REWRITE_PROMPT.format(question=question)
        rewritten = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert academic research query optimizer. Return only the rewritten query.",
            temperature=0.2
        )
        cleaned = rewritten.strip().strip('"').strip("'")
        if not cleaned or len(cleaned) < 5 or cleaned.startswith("[Error"):
            # Fallback: append domain search terms
            return f"{question} object detection computer vision algorithm methodology"
        return cleaned
