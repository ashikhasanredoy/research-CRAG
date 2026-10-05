import re
import json
import logging
from typing import List, Dict, Any, Tuple
from app.llm.ollama_client import OllamaClient
from app.config import settings

logger = logging.getLogger(__name__)

class RetrievalGrader:
    """Evaluates whether retrieved paper chunks are relevant to the user query."""

    BATCH_GRADER_PROMPT = """You are an academic retrieval evaluator. Determine whether each document is relevant to the question.

Question: {question}

Candidate Documents:
{documents_text}

Respond ONLY in this exact JSON format:
{{"results": [{{"id": 1, "relevant": true}}, {{"id": 2, "relevant": false}}]}}
"""

    def __init__(self, llm_client: OllamaClient = None):
        self.llm = llm_client or OllamaClient()

    def grade_documents(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Grades all candidate documents in a single fast batch LLM request.
        Returns: (relevant_docs, irrelevant_docs, status ['good' | 'bad'])
        """
        if not documents:
            return [], [], "bad"

        # Build formatted candidate list
        docs_text_blocks = []
        for i, doc in enumerate(documents, start=1):
            snippet = doc.get("text", "")[:400].replace("\n", " ")
            meta = doc.get("metadata", {})
            paper = meta.get("paper", "")
            docs_text_blocks.append(f"[{i}] (Paper: {paper}): \"{snippet}\"")

        prompt = self.BATCH_GRADER_PROMPT.format(
            question=question,
            documents_text="\n".join(docs_text_blocks)
        )

        relevant_indices = set()
        reasons_map = {}
        try:
            response_text = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a strict academic retrieval evaluator. Output ONLY valid JSON in the format {\"evaluations\": [{\"id\": 1, \"relevant\": true, \"reason\": \"...\"}]}",
                format_json=False,
                temperature=0.0,
                max_tokens=150
            )

            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            evals = parsed.get("evaluations", parsed.get("results", []))
            
            for item in evals:
                doc_id = item.get("id")
                rel = item.get("relevant")
                if isinstance(rel, str):
                    rel = "yes" in rel.lower() or "true" in rel.lower()
                if rel:
                    relevant_indices.add(doc_id)
                reasons_map[doc_id] = item.get("reason", "Graded by batch evaluator")

        except Exception as e:
            logger.warning(f"Batch grader fallback triggered ({e}). Using semantic keyword overlap heuristic.")
            stopwords = {"the", "a", "an", "in", "on", "of", "and", "or", "to", "for", "is", "are", "with", "by", "what", "how", "why", "who", "where", "when", "can", "i", "my", "me", "you", "your", "feel", "pain", "head"}
            q_terms = [w.lower() for w in re.findall(r'\b\w+\b', question) if w.lower() not in stopwords and len(w) > 2]

            for idx, doc in enumerate(documents, start=1):
                text_lower = doc.get("text", "").lower()
                matching_terms = [w for w in q_terms if w in text_lower]
                overlap = len(matching_terms) / max(len(q_terms), 1) if q_terms else 0.0

                if overlap >= 0.30:
                    relevant_indices.add(idx)
                    reasons_map[idx] = f"Keyword overlap match ({int(overlap*100)}%)"

        relevant_docs: List[Dict[str, Any]] = []
        irrelevant_docs: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents, start=1):
            doc_graded = dict(doc)
            is_rel = idx in relevant_indices
            doc_graded["grader_verdict"] = "relevant" if is_rel else "irrelevant"
            doc_graded["grader_reason"] = reasons_map.get(idx, "Not relevant")

            if is_rel:
                relevant_docs.append(doc_graded)
            else:
                irrelevant_docs.append(doc_graded)

        # Retrieval is GOOD if at least 1 document is relevant
        status = "good" if len(relevant_docs) > 0 else "bad"
        return relevant_docs, irrelevant_docs, status
