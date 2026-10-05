import logging
from typing import List, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

class WebSearchFallback:
    """Performs web search fallback using DuckDuckGo when local research papers lack sufficient context."""

    def __init__(self, max_results: int = 4):
        self.max_results = max_results

    SPAM_KEYWORDS = {
        "baji", "casino", "slots", "slot game", "baccarat", "betting", "jackpot",
        "1xbet", "poker", "roulette", "bonus", "gambling", "sexy baccarat"
    }

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Searches DuckDuckGo / ddgs with worldwide quality filter and formats results as document chunks."""
        if not settings.ENABLE_WEB_SEARCH_FALLBACK:
            return []

        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            raw_results = []
            with DDGS() as ddgs:
                # Use worldwide region (wt-wt) and fetch extra candidates to filter spam
                raw_results = list(ddgs.text(query, region="wt-wt", max_results=self.max_results * 2))

            web_docs: List[Dict[str, Any]] = []
            for i, res in enumerate(raw_results):
                title = res.get("title", f"Web Source {i+1}")
                body = res.get("body", "")
                href = res.get("href", "")

                if not body or len(body.strip()) < 20:
                    continue

                # Filter out gambling / spam results
                text_check = f"{title} {body} {href}".lower()
                if any(spam in text_check for spam in self.SPAM_KEYWORDS):
                    continue

                web_docs.append({
                    "id": f"web_search_{len(web_docs)}",
                    "text": f"{title}\n{body}",
                    "metadata": {
                        "paper": title[:40],
                        "paper_title": title,
                        "page": 1,
                        "section": "Web Search Knowledge",
                        "url": href,
                        "chunk_id": len(web_docs)
                    },
                    "source_type": "web_search",
                    "score": 0.8
                })

                if len(web_docs) >= self.max_results:
                    break

            logger.info(f"Web search retrieved {len(web_docs)} high-quality web documents for query: {query}")
            return web_docs

        except Exception as e:
            logger.warning(f"Web search fallback failed ({e}). Continuing with available local context.")
            return []
