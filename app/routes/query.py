"""
Module C: RAG query endpoint.

POST /api/v1/query
  Body: { "project_id": str, "query": str }

Pipeline:
  1. ChromaDB embeds the query and performs cosine similarity search
     → returns top-K most relevant chunks for the project
  2. Chunks stuffed as context into a Groq LLM call
  3. Returns a factual, grounded answer
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.chroma import query_chunks
from app.services.llm import generate_rag_answer

router = APIRouter()


class QueryRequest(BaseModel):
    project_id: str = Field(..., description="Project to query against")
    query: str = Field(..., min_length=3, description="Natural language question")


class QueryResponse(BaseModel):
    project_id: str
    query: str
    answer: str
    source_chunks: list[str]
    chunks_used: int


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query project requirements via RAG",
    description=(
        "ChromaDB performs cosine similarity search over the project's stored chunks. "
        "Top-K results are fed to Groq to generate a factual, grounded answer."
    ),
)
def query(payload: QueryRequest) -> QueryResponse:

    # ── Step 1: Vector retrieval ─────────────────────────────────────────────
    try:
        chunks = query_chunks(
            project_id=payload.project_id,
            query=payload.query,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector retrieval failed: {exc}",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No chunks found for project_id='{payload.project_id}'. "
                "Make sure you have ingested a transcript for this project first."
            ),
        )

    # ── Step 2: LLM answer generation ───────────────────────────────────────
    try:
        answer = generate_rag_answer(payload.query, chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM answer generation failed: {exc}",
        ) from exc

    return QueryResponse(
        project_id=payload.project_id,
        query=payload.query,
        answer=answer,
        source_chunks=chunks,
        chunks_used=len(chunks),
    )
