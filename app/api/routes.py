import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel

from app.config import settings
from app.ingestion.pdf_loader import PDFPaperLoader
from app.ingestion.chunker import ResearchPaperChunker
from app.vectorstore.chroma import ChromaVectorStoreManager
from app.retrieval.retriever import HybridRetriever
from app.crag.graph import CRAGWorkflow
from app.llm.ollama_client import OllamaClient
from app.evaluation.evaluator import RAGEvaluator

router = APIRouter()

# Global instances
vectorstore = ChromaVectorStoreManager()
retriever = HybridRetriever(vectorstore=vectorstore)
crag_pipeline = CRAGWorkflow()
ollama_client = OllamaClient()
evaluator = RAGEvaluator(crag_workflow=crag_pipeline, retriever=retriever)

class QueryRequest(BaseModel):
    question: str
    llm_model: Optional[str] = None
    temperature: Optional[float] = None
    enable_web_search: Optional[bool] = None

class EvaluateRequest(BaseModel):
    question: str

class SettingsUpdateRequest(BaseModel):
    llm_model: Optional[str] = None
    temperature: Optional[float] = None
    use_reranker: Optional[bool] = None
    enable_web_search: Optional[bool] = None

@router.get("/health")
def get_health() -> Dict[str, Any]:
    """Returns system status, Ollama availability, and ChromaDB stats."""
    ollama_ok = ollama_client.check_health()
    local_models = ollama_client.list_local_models() if ollama_ok else []
    indexed_count = vectorstore.count()

    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "ollama": {
            "status": "connected" if ollama_ok else "unreachable",
            "base_url": settings.OLLAMA_BASE_URL,
            "active_llm": settings.DEFAULT_LLM_MODEL,
            "active_embed": settings.DEFAULT_EMBED_MODEL,
            "available_models": local_models
        },
        "vectorstore": {
            "type": "ChromaDB",
            "total_chunks": indexed_count,
            "persist_dir": str(settings.CHROMA_PERSIST_DIR)
        }
    }

from fastapi.responses import StreamingResponse
import json

@router.post("/query")
def process_query(request: QueryRequest) -> Dict[str, Any]:
    """Executes the Corrective RAG (CRAG) pipeline for a user question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.llm_model:
        ollama_client.model_name = request.llm_model

    result = crag_pipeline.run(question=request.question)
    return {
        "question": request.question,
        "original_question": result.get("original_question"),
        "rewritten_query": result.get("rewritten_query"),
        "rewrite_count": result.get("rewrite_count", 0),
        "retrieval_status": result.get("retrieval_status"),
        "confidence_level": result.get("confidence_level", "LOW"),
        "retrieval_confidence": result.get("retrieval_confidence", 0.0),
        "fast_path_used": result.get("fast_path_used", False),
        "answer": result.get("generation"),
        "citations": result.get("citations", []),
        "verification": result.get("verification_result", {}),
        "graded_documents": [
            {
                "id": doc.get("id"),
                "paper": doc.get("metadata", {}).get("paper"),
                "page": doc.get("metadata", {}).get("page"),
                "section": doc.get("metadata", {}).get("section"),
                "verdict": doc.get("grader_verdict", "relevant"),
                "reason": doc.get("grader_reason", "Confidence fast-path"),
                "score": doc.get("rerank_score") or doc.get("hybrid_score") or doc.get("score")
            }
            for doc in result.get("graded_documents", [])
        ],
        "web_search_used": result.get("web_search_used", False),
        "trace": result.get("trace", [])
    }

@router.post("/query/stream")
def process_query_stream(request: QueryRequest):
    """Streams response tokens in real-time with step trace and citations."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.llm_model:
        ollama_client.model_name = request.llm_model

    def event_stream():
        nodes = crag_pipeline.nodes
        state = {
            "question": request.question,
            "original_question": request.question,
            "cleaned_query": request.question,
            "rewritten_query": None,
            "rewrite_count": 0,
            "raw_documents": [],
            "retrieval_confidence": 0.0,
            "confidence_level": "LOW",
            "fast_path_used": False,
            "graded_documents": [],
            "irrelevant_documents": [],
            "retrieval_status": "pending",
            "context": "",
            "generation": "",
            "verification_result": {},
            "citations": [],
            "web_search_used": False,
            "trace": []
        }

        # Step 1: Query Preprocessing
        state.update(nodes.preprocess_query(state))

        # Step 2: Hybrid Retrieval (Vector + BM25)
        state.update(nodes.retrieve_documents(state))

        # Step 3: Retrieval Confidence Estimator
        state.update(nodes.estimate_confidence(state))

        yield f"data: {json.dumps({'type': 'status', 'stage': 'confidence', 'confidence': state.get('retrieval_confidence'), 'level': state.get('confidence_level'), 'fast_path': state.get('fast_path_used')})}\n\n"

        # Step 4: Fast Path vs LLM Grader
        if state.get("confidence_level") == "HIGH" or state.get("fast_path_used"):
            state.update(nodes.aggregate_context(state))
        else:
            yield f"data: {json.dumps({'type': 'status', 'stage': 'llm_grader'})}\n\n"
            state.update(nodes.grade_documents(state))

            if state.get("retrieval_status") == "good":
                state.update(nodes.aggregate_context(state))
            else:
                # Step 5: Query Rewriter (LLM Call #2)
                yield f"data: {json.dumps({'type': 'status', 'stage': 'query_rewriter'})}\n\n"
                state.update(nodes.rewrite_query(state))
                state.update(nodes.re_retrieve_and_check(state))

                if state.get("retrieval_status") == "good":
                    state.update(nodes.aggregate_context(state))
                elif settings.ENABLE_WEB_SEARCH_FALLBACK:
                    # Step 6: Web Search Fallback
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'web_search'})}\n\n"
                    state.update(nodes.web_search_node(state))
                    state.update(nodes.aggregate_context(state))
                else:
                    state.update(nodes.aggregate_context(state))

        # Step 7: Answer Generator (LLM Call #3) Token Streaming
        docs = state.get("graded_documents", [])
        stream_gen, citations, context_str = nodes.generator.generate_stream(
            question=state.get("original_question") or state["question"],
            documents=docs
        )
        state["citations"] = citations
        state["context"] = context_str

        full_tokens = []
        for token in stream_gen:
            full_tokens.append(token)
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        full_text = "".join(full_tokens)
        state["generation"] = full_text

        # Step 8: Citation / Source Verification
        state.update(nodes.verify_citations(state))

        # Done event
        yield f"data: {json.dumps({'type': 'done', 'answer': full_text, 'citations': state.get('citations', []), 'trace': state.get('trace', []), 'rewritten_query': state.get('rewritten_query'), 'fast_path_used': state.get('fast_path_used', False), 'confidence_level': state.get('confidence_level', 'LOW'), 'retrieval_confidence': state.get('retrieval_confidence', 0.0), 'verification': state.get('verification_result', {})})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/upload")
async def upload_paper(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Uploads a research paper PDF and indexes it into ChromaDB vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = settings.PAPERS_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Extract paper pages and sections
        loader = PDFPaperLoader(save_path)
        paper_data = loader.extract_paper()

        # 2. Chunk paper
        chunker = ResearchPaperChunker()
        chunks = chunker.chunk_paper(paper_data)

        # 3. Store in ChromaDB
        indexed_count = vectorstore.add_chunks(chunks)

        # 4. Refresh retriever BM25 index
        retriever.refresh_bm25_index()

        return {
            "message": f"Successfully indexed '{file.filename}'",
            "title": paper_data.get("title"),
            "total_pages": paper_data.get("total_pages"),
            "chunks_created": indexed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {str(e)}")

@router.get("/papers")
def list_papers() -> List[Dict[str, Any]]:
    """Returns list of all indexed research papers and metadata."""
    return vectorstore.list_indexed_papers()

@router.delete("/papers/{paper_name}")
def delete_paper(paper_name: str) -> Dict[str, Any]:
    """Deletes an indexed paper and its chunks from ChromaDB."""
    vectorstore.delete_paper(paper_name)
    retriever.refresh_bm25_index()
    
    # Remove physical file if present
    file_path = settings.PAPERS_DIR / paper_name
    if file_path.exists():
        file_path.unlink()

    return {"message": f"Deleted paper '{paper_name}' successfully."}

@router.post("/evaluate")
def evaluate_query(request: EvaluateRequest) -> Dict[str, Any]:
    """Runs a side-by-side comparison between Basic RAG and CRAG on a test question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return evaluator.compare(request.question)

@router.get("/settings")
def get_settings() -> Dict[str, Any]:
    """Returns current system configuration settings."""
    return {
        "default_llm": settings.DEFAULT_LLM_MODEL,
        "default_embed": settings.DEFAULT_EMBED_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "use_reranker": settings.USE_RERANKER,
        "web_search_fallback": settings.ENABLE_WEB_SEARCH_FALLBACK,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "reranker_top_k": settings.RERANKER_TOP_K
    }
