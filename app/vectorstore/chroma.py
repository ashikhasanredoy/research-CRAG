import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.embeddings.embedding_model import get_embedding_model

logger = logging.getLogger(__name__)

class ChromaVectorStoreManager:
    """Manages persistent ChromaDB vector store collection for research papers."""

    COLLECTION_NAME = "research_papers"

    def __init__(
        self,
        persist_directory: str = str(settings.CHROMA_PERSIST_DIR),
        embedding_model_name: str = settings.DEFAULT_EMBED_MODEL
    ):
        self.persist_directory = persist_directory
        self.embedding_service = get_embedding_model(model_name=embedding_model_name)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Adds chunks to ChromaDB vector store with duplicate prevention."""
        if not chunks:
            return 0

        # Extract texts, ids, metadatas
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generate embeddings
        embeddings = self.embedding_service.embed_documents(texts)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Successfully indexed {len(chunks)} chunks in ChromaDB.")
        return len(chunks)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = settings.RETRIEVAL_TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Performs vector similarity search and returns documents with scores and metadata."""
        query_embedding = self.embedding_service.embed_query(query)
        
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata if filter_metadata else None,
            include=["documents", "metadatas", "distances"]
        )

        matched_docs: List[Dict[str, Any]] = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                # In cosine distance: score = 1 - distance
                cosine_dist = distances[i]
                sim_score = max(0.0, 1.0 - cosine_dist)
                matched_docs.append({
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "score": round(sim_score, 4),
                    "source_type": "vector"
                })

        return matched_docs

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieves all indexed documents (used for building BM25 index)."""
        count = self.collection.count()
        if count == 0:
            return []
        
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        docs = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                docs.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
        return docs

    def list_indexed_papers(self) -> List[Dict[str, Any]]:
        """Returns aggregated information for all indexed papers."""
        all_docs = self.get_all_documents()
        papers_map: Dict[str, Dict[str, Any]] = {}

        for doc in all_docs:
            meta = doc.get("metadata", {})
            p_name = meta.get("paper", "Unknown")
            p_title = meta.get("paper_title", p_name)
            p_page = meta.get("page", 1)

            if p_name not in papers_map:
                papers_map[p_name] = {
                    "filename": p_name,
                    "title": p_title,
                    "chunk_count": 0,
                    "max_page": 1,
                    "sections": set()
                }

            papers_map[p_name]["chunk_count"] += 1
            papers_map[p_name]["max_page"] = max(papers_map[p_name]["max_page"], p_page)
            if "section" in meta and meta["section"]:
                papers_map[p_name]["sections"].add(meta["section"])

        # Format output
        paper_list = []
        for p_name, info in papers_map.items():
            paper_list.append({
                "filename": info["filename"],
                "title": info["title"],
                "chunk_count": info["chunk_count"],
                "total_pages": info["max_page"],
                "sections": list(info["sections"])
            })
        return paper_list

    def delete_paper(self, filename: str) -> int:
        """Deletes all chunks associated with a specific paper filename."""
        self.collection.delete(where={"paper": filename})
        logger.info(f"Deleted all chunks for paper: {filename}")
        return self.collection.count()

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """Clears all vectors in collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
