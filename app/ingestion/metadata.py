import hashlib
from typing import Dict, Any

def generate_chunk_hash(text: str, paper_name: str, page: int, chunk_id: int) -> str:
    """Creates a deterministic hash for a document chunk to prevent duplicates."""
    raw = f"{paper_name}::{page}::{chunk_id}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def build_chunk_metadata(
    paper_name: str,
    paper_title: str,
    page_number: int,
    section: str,
    chunk_id: int,
    text: str
) -> Dict[str, Any]:
    """Builds clean, sanitized metadata dictionary suitable for ChromaDB vector store."""
    chunk_hash = generate_chunk_hash(text, paper_name, page_number, chunk_id)
    return {
        "paper": paper_name,
        "paper_title": paper_title,
        "page": int(page_number),
        "section": section or "General",
        "chunk_id": int(chunk_id),
        "chunk_hash": chunk_hash
    }
