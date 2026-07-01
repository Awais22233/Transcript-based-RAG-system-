"""
Project management endpoints.

GET    /api/v1/projects              — list all projects
GET    /api/v1/projects/{project_id} — project detail with full PRD markdown
DELETE /api/v1/projects/{project_id} — delete project, documents, and ChromaDB chunks
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models import Project, Document
from app.services.chroma import delete_project_chunks

router = APIRouter()


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    document_count: int
    total_chunks: int
    created_at: str


class DocumentOut(BaseModel):
    id: int
    chunk_count: int
    markdown_content: str
    created_at: str


class ProjectDetail(BaseModel):
    project_id: str
    name: str
    created_at: str
    documents: list[DocumentOut]


@router.get(
    "/projects",
    response_model=list[ProjectSummary],
    summary="List all projects",
)
def list_projects(db: Session = Depends(get_db)) -> list[ProjectSummary]:
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    result = []
    for p in projects:
        doc_count = (
            db.query(func.count(Document.id))
            .filter(Document.project_id == p.project_id)
            .scalar() or 0
        )
        total_chunks = (
            db.query(func.sum(Document.chunk_count))
            .filter(Document.project_id == p.project_id)
            .scalar() or 0
        )
        result.append(
            ProjectSummary(
                project_id=p.project_id,
                name=p.name,
                document_count=doc_count,
                total_chunks=total_chunks,
                created_at=p.created_at.isoformat(),
            )
        )
    return result


@router.get(
    "/projects/{project_id}",
    response_model=ProjectDetail,
    summary="Get project detail with full PRD markdown",
)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectDetail:
    project = db.query(Project).filter_by(project_id=project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )
    docs = (
        db.query(Document)
        .filter(Document.project_id == project_id)
        .order_by(Document.created_at)
        .all()
    )
    return ProjectDetail(
        project_id=project.project_id,
        name=project.name,
        created_at=project.created_at.isoformat(),
        documents=[
            DocumentOut(
                id=d.id,
                chunk_count=d.chunk_count,
                markdown_content=d.markdown_content,
                created_at=d.created_at.isoformat(),
            )
            for d in docs
        ],
    )


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a project and all its data",
)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> None:
    project = db.query(Project).filter_by(project_id=project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )
    # Delete from SQLite (cascades to documents)
    db.delete(project)
    db.commit()
    # Delete chunks from ChromaDB
    try:
        delete_project_chunks(project_id)
    except Exception:
        pass  # SQLite delete already succeeded; ChromaDB cleanup is best-effort
