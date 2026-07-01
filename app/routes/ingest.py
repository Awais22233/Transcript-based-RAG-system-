"""
Module A + B: Ingest endpoint.

POST /api/v1/ingest
  Body: { "project_id": str, "project_name": str, "transcript": str }

Pipeline:
  1. Groq LLM transforms raw transcript → structured Markdown PRD
  2. PRD saved to SQLite `documents` table (linked to project)
  3. PRD chunked (512-token window, 51-token overlap)
  4. Chunks stored in ChromaDB (embedding handled automatically)
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Project, Document
from app.services.llm import transform_transcript
from app.services.chunker import chunk_markdown
from app.services.chroma import add_chunks

router = APIRouter()


class IngestRequest(BaseModel):
    project_id: str = Field(..., description="Unique project identifier (e.g. 'proj-001')")
    project_name: str = Field(..., description="Human-readable project name")
    transcript: str = Field(..., min_length=10, description="Raw speaker-diarized transcript text")


class IngestResponse(BaseModel):
    project_id: str
    project_name: str
    document_id: int
    chunk_count: int
    markdown_preview: str   # first 500 chars of the generated PRD


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a meeting transcript",
    description=(
        "Sends the raw transcript to Groq LLM which produces a structured Markdown PRD. "
        "The PRD is saved to SQLite, then chunked and stored in ChromaDB for RAG queries."
    ),
)
def ingest(payload: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:

    # ── Step 1: LLM transformation ──────────────────────────────────────────
    try:
        markdown_prd = transform_transcript(payload.transcript)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM transformation failed: {exc}",
        ) from exc

    # ── Step 2: Save to SQLite (atomic) ─────────────────────────────────────
    try:
        # Upsert project
        project = db.query(Project).filter_by(project_id=payload.project_id).first()
        if project is None:
            project = Project(
                project_id=payload.project_id,
                name=payload.project_name,
            )
            db.add(project)
            db.flush()

        # Chunk the PRD
        chunks = chunk_markdown(markdown_prd)

        # Save document row
        document = Document(
            project_id=payload.project_id,
            markdown_content=markdown_prd,
            chunk_count=len(chunks),
        )
        db.add(document)
        db.flush()  # get document.id before committing

        db.commit()
        db.refresh(document)

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database write failed: {exc}",
        ) from exc

    # ── Step 3: Store chunks in ChromaDB ────────────────────────────────────
    # Done outside the SQLite transaction — ChromaDB is not transactional,
    # but upsert is idempotent so retries are safe.
    try:
        add_chunks(
            project_id=payload.project_id,
            document_id=document.id,
            chunks=chunks,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ChromaDB write failed: {exc}",
        ) from exc

    return IngestResponse(
        project_id=payload.project_id,
        project_name=payload.project_name,
        document_id=document.id,
        chunk_count=len(chunks),
        markdown_preview=markdown_prd[:500],
    )
