"""
Token-based markdown chunker (Module B).

Strategy: sliding window over tiktoken tokens.
  - window  : CHUNK_TOKENS         (default 512)
  - overlap : CHUNK_OVERLAP_TOKENS  (default 51, ~10%)
"""
from __future__ import annotations

import tiktoken
from app.config import settings

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_markdown(markdown: str) -> list[str]:
    """
    Split *markdown* into overlapping token-window chunks.
    Returns at least one chunk even for very short documents.
    """
    if settings.chunk_overlap_tokens >= settings.chunk_tokens:
        raise ValueError(
            f"CHUNK_OVERLAP_TOKENS ({settings.chunk_overlap_tokens}) must be "
            f"less than CHUNK_TOKENS ({settings.chunk_tokens})"
        )

    tokens = _ENCODING.encode(markdown)
    window = settings.chunk_tokens
    overlap = settings.chunk_overlap_tokens
    step = window - overlap

    if len(tokens) <= window:
        return [markdown]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + window, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step

    return chunks
