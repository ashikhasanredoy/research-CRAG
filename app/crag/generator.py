import logging
from typing import List, Dict, Any, Tuple
from app.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class GroundedAnswerGenerator:
    """Generates faithful, citation-grounded research answers strictly based on retrieved context."""

    SYSTEM_PROMPT = """You are ResearchCRAG, an AI research assistant for AI/ML computer vision papers.
Rules:
1. Answer the question directly and concisely using ONLY the provided Context.
2. Cite sources in brackets like [PaperName, Page X] for every key finding or claim.
3. Keep the answer clear, structured, and focused (2-4 concise paragraphs or bullet points).
4. Do not invent information not in the context.
"""

    USER_PROMPT = """Question: {question}

Context from Papers:
{context}

Provide a direct, concise, grounded answer with citations:"""

    def __init__(self, llm_client: OllamaClient = None):
        self.llm = llm_client or OllamaClient()

    def build_context(self, documents: List[Dict[str, Any]], max_docs: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Formats top document chunks into an organized context string and collects citation metadata."""
        if not documents:
            return "No relevant documents found.", []

        context_blocks = []
        citations = []
        seen_citations = set()

        for i, doc in enumerate(documents[:max_docs], start=1):
            meta = doc.get("metadata", {})
            paper = meta.get("paper", "Unknown Paper")
            title = meta.get("paper_title", paper)
            page = meta.get("page", 1)
            section = meta.get("section", "General")
            text = doc.get("text", "").strip()
            # Trim excessively long text to prevent LLM prompt stalling
            trimmed_text = text[:1000]

            block = f"--- Document [{i}] ---\n"
            block += f"Paper: {title} ({paper})\n"
            block += f"Page: {page} | Section: {section}\n"
            block += f"Content:\n{trimmed_text}\n"
            context_blocks.append(block)

            cit_key = f"{paper}::p{page}"
            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations.append({
                    "paper": paper,
                    "paper_title": title,
                    "page": page,
                    "section": section,
                    "snippet": trimmed_text[:180] + "..." if len(trimmed_text) > 180 else trimmed_text,
                    "url": meta.get("url")
                })

        return "\n".join(context_blocks), citations

    def generate(self, question: str, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], str]:
        """Generates answer from documents and returns (answer, citations, context_str)."""
        context_str, citations = self.build_context(documents)
        prompt = self.USER_PROMPT.format(question=question, context=context_str)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=450
        )
        return response, citations, context_str

    def generate_stream(self, question: str, documents: List[Dict[str, Any]]):
        """Streams tokens from LLM given the aggregated context."""
        context_str, citations = self.build_context(documents)
        prompt = self.USER_PROMPT.format(question=question, context=context_str)
        stream = self.llm.generate_stream(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1
        )
        return stream, citations, context_str
