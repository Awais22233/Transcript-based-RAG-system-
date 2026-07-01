"""
ChromaDB service — handles all vector storage and similarity search.

One persistent ChromaDB collection ("requirements") stores every chunk
across all projects. project_id is stored as metadata so queries are
always scoped to a single project.

ChromaDB handles embedding automatically via its built-in
SentenceTransformerEmbeddingFunction — no separate embed step needed.
"""
from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

COLLECTION_NAME = "requirements"


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_db_path)


@lru_cache(maxsize=1)
def _get_embedding_fn() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )


def get_collection() -> chromadb.Collection:
    """Return (or create) the shared collection with cosine similarity."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def add_chunks(project_id: str, document_id: int, chunks: list[str]) -> None:
    """
    Store text chunks in ChromaDB.
    IDs are deterministic: "{project_id}__{document_id}__{chunk_index}"
    so re-ingesting the same document replaces old chunks cleanly.
    """
    collection = get_collection()

    ids = [f"{project_id}__{document_id}__{i}" for i in range(len(chunks))]
    metadatas = [
        {"project_id": project_id, "document_id": str(document_id), "chunk_index": i}
        for i in range(len(chunks))
    ]

    # Upsert — safe to call multiple times for the same document
    collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)


def delete_project_chunks(project_id: str) -> None:
    """Remove every chunk that belongs to project_id."""
    collection = get_collection()
    results = collection.get(where={"project_id": project_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def query_chunks(project_id: str, query: str, top_k: int | None = None) -> list[str]:
    """
    Embed *query* and return the top-K most similar chunk texts for *project_id*.
    Returns an empty list if no chunks exist for the project.
    """
    k = top_k or settings.top_k_chunks
    collection = get_collection()

    # How many chunks exist for this project?
    existing = collection.get(where={"project_id": project_id})
    total = len(existing["ids"])
    if total == 0:
        return []

    # ChromaDB errors if n_results > total documents in collection
    n = min(k, total)

    results = collection.query(
        query_texts=[query],
        n_results=n,
        where={"project_id": project_id},
    )

    docs = results.get("documents", [[]])[0]
    return docs if docs else []
