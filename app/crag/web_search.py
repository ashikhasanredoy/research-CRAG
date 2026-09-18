import logging
from typing import List, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

class WebSearchFallback:
    """Performs web search fallback using DuckDuckGo when local research papers lack sufficient context."""

    def __init__(self, max_results: int = 4):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Searches DuckDuckGo and formats results as document chunks."""
        if not settings.ENABLE_WEB_SEARCH_FALLBACK:
            return []

        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                academic_query = f"{query} research paper arXiv"
                raw_results = list(ddgs.text(academic_query, max_results=self.max_results))

            web_docs: List[Dict[str, Any]] = []
            for i, res in enumerate(raw_results):
                title = res.get("title", f"Web Source {i+1}")
                body = res.get("body", "")
                href = res.get("href", "")

                if not body:
                    continue

                web_docs.append({
                    "id": f"web_search_{i}",
                    "text": f"Title: {title}\nSummary: {body}\nURL: {href}",
                    "metadata": {
                        "paper": f"Web: {title[:40]}...",
                        "paper_title": title,
                        "page": 1,
                        "section": "Web Knowledge Fallback",
                        "url": href,
                        "chunk_id": i
                    },
                    "source_type": "web_search",
                    "score": 0.8
                })

            logger.info(f"Web search retrieved {len(web_docs)} web documents for query: {query}")
            return web_docs

        except Exception as e:
            logger.warning(f"Web search fallback failed ({e}). Continuing with available local context.")
            return []
