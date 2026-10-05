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

    def _fallback_synthesis(self, question: str, documents: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> str:
        """Fallback extractive synthesizer when Ollama is unavailable or returns an error."""
        if not documents:
            return "Based on the indexed AI/ML research papers, no directly matching context was found for this query."
        
        # Check if any document is from web search
        has_web = any(d.get("source_type") == "web_search" for d in documents)
        
        paragraphs = []
        if has_web:
            for doc in documents[:2]:
                if doc.get("source_type") == "web_search":
                    meta = doc.get("metadata", {})
                    title = meta.get("paper_title", "Web Source")
                    text = doc.get("text", "").strip()
                    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
                    summary_snippet = " ".join(lines[:3]) if lines else text[:300]
                    paragraphs.append(f"According to **{title}** [{title}, Page 1]: {summary_snippet}")
            return "\n\n".join(paragraphs) if paragraphs else "Found relevant web information for your query."

        # For local papers, check keyword relevance
        import re
        stopwords = {"the", "a", "an", "in", "on", "of", "and", "or", "to", "for", "is", "are", "with", "by", "what", "how", "why", "who", "where", "when", "can", "i", "my", "me", "you", "your", "feel", "pain", "head"}
        q_terms = [w.lower() for w in re.findall(r'\b\w+\b', question) if w.lower() not in stopwords and len(w) > 2]
        
        relevant_docs = []
        for doc in documents:
            text_lower = doc.get("text", "").lower()
            if any(term in text_lower for term in q_terms):
                relevant_docs.append(doc)

        if not relevant_docs:
            return "This query is outside the domain of the indexed AI/ML computer vision papers (IA-YOLO, PE-YOLO, AOD-Net, CRAG). No relevant findings were located."

        for doc in relevant_docs[:2]:
            meta = doc.get("metadata", {})
            title = meta.get("paper_title", meta.get("paper", "Paper"))
            page = meta.get("page", 1)
            section = meta.get("section", "General")
            text = doc.get("text", "").strip()
            lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 30]
            summary_snippet = " ".join(lines[:3]) if lines else text[:300]
            paragraphs.append(f"According to **{title}** (Section: {section}) [{title}, Page {page}]: {summary_snippet}")
        
        return "\n\n".join(paragraphs)

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
        if response.startswith("[Error:"):
            logger.info("Ollama error encountered during answer generation. Using fallback synthesis.")
            response = self._fallback_synthesis(question, documents, citations)

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
        
        def safe_stream():
            collected = []
            has_error = False
            for chunk in stream:
                if chunk.startswith("[Error:"):
                    has_error = True
                    break
                collected.append(chunk)
                yield chunk
            
            if has_error:
                fallback_text = self._fallback_synthesis(question, documents, citations)
                # If nothing was yielded yet, stream fallback words
                if not collected:
                    import time
                    for word in fallback_text.split(" "):
                        yield word + " "
                        time.sleep(0.02)

        return safe_stream(), citations, context_str
