"""
Ingests all PDF files in data/papers/ into the ChromaDB vector store.
"""
from pathlib import Path
from app.config import settings
from app.ingestion.pdf_loader import PDFPaperLoader
from app.ingestion.chunker import ResearchPaperChunker
from app.vectorstore.chroma import ChromaVectorStoreManager
from app.retrieval.retriever import HybridRetriever

def ingest_all():
    print(f"Scanning papers directory: {settings.PAPERS_DIR}")
    pdf_files = list(settings.PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/papers/.")
        return

    vectorstore = ChromaVectorStoreManager()
    chunker = ResearchPaperChunker()
    total_indexed = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}...")
        loader = PDFPaperLoader(pdf_path)
        paper_data = loader.extract_paper()
        chunks = chunker.chunk_paper(paper_data)
        count = vectorstore.add_chunks(chunks)
        total_indexed += count
        print(f"  -> Extracted {paper_data['total_pages']} pages, indexed {count} chunks.")

    retriever = HybridRetriever(vectorstore=vectorstore)
    retriever.refresh_bm25_index()
    print(f"\nCompleted! Total chunks indexed across all papers: {total_indexed}")

if __name__ == "__main__":
    ingest_all()
