from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.metadata import build_chunk_metadata
from app.config import settings

class ResearchPaperChunker:
    """Chunks academic paper text by recursive character splitting with page & section preservation."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n[Section: ", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def chunk_paper(self, paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Takes output of PDFPaperLoader and returns list of chunks with complete metadata."""
        paper_name = paper_data["filename"]
        paper_title = paper_data.get("title", paper_name)
        chunks: List[Dict[str, Any]] = []
        global_chunk_id = 0

        for page in paper_data.get("pages", []):
            page_num = page["page_number"]
            page_text = page["text"]
            page_section = page.get("section", "General")

            if not page_text.strip():
                continue

            page_chunks = self.splitter.split_text(page_text)
            for raw_chunk in page_chunks:
                clean_chunk = raw_chunk.strip()
                if len(clean_chunk) < 40:  # Skip tiny fragments
                    continue

                # Check if this specific chunk has an explicit section marker
                current_section = page_section
                if "[Section: " in clean_chunk:
                    try:
                        extracted_sec = clean_chunk.split("[Section: ")[1].split("]")[0]
                        if extracted_sec:
                            current_section = extracted_sec
                    except IndexError:
                        pass

                meta = build_chunk_metadata(
                    paper_name=paper_name,
                    paper_title=paper_title,
                    page_number=page_num,
                    section=current_section,
                    chunk_id=global_chunk_id,
                    text=clean_chunk
                )

                chunks.append({
                    "id": f"{paper_name}_p{page_num}_c{global_chunk_id}",
                    "text": clean_chunk,
                    "metadata": meta
                })
                global_chunk_id += 1

        return chunks
