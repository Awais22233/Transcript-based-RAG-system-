"""
Groq LLM service.
  - transform_transcript()  : raw transcript → structured Markdown PRD
  - generate_rag_answer()   : retrieved chunks + query → concise answer
"""
from groq import Groq
from app.config import settings

_client = Groq(api_key=settings.groq_api_key)

# ---------------------------------------------------------------------------
# System prompt — exact wording from Section 4 of the PRD
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Systems Analyst and Senior Requirements Engineer.
Your task is to analyze raw meeting transcripts from project discovery calls and synthesize them into a clean, highly structured, and technical Product Requirements Document (PRD) written in Markdown.

### Rules of Engagement:
1. Extract every explicit developer constraint, client preference, functional feature request, and business logic mentioned.
2. Filter out conversational noise, small talk, filler words, and meeting administrative chatter.
3. Be technically precise. If UI details are discussed (e.g., specific colors like emerald green and white, or layouts), explicitly log them in a dedicated "UI/UX & Aesthetics" section.
4. Maintain a structured, professional, and authoritative tone. Use bullet points and clean headers.
5. If something critical to the technical implementation is ambiguous, list it explicitly in an "Outstanding Queries / Assumptions" section at the bottom.

### Output Schema:
Your output must strictly follow this structural layout:

# Project Name: [Extract Name or Provide Descriptive Working Title]
## 1. Project Summary & Core Objective
[Brief paragraph summarizing what the client is building and why]

## 2. Functional Requirements & Feature Sets
### [Module Name / Epic Name]
* **Requirement ID:** FR-00X
* **Description:** [Clear description of what the system must do]
* **User Flow Context:** [Who interacts with this and how based on the call]

## 3. Non-Functional & Technical Requirements
* **Performance/Scale:** [Any mentions of users, speed, or platform choices]
* **Security/Auth:** [Authentication, role-based access control mentions]

## 4. UI/UX, Design, & Aesthetic Constraints
* **Color Palette:** [Log specific hex codes or exact color terms used]
* **Layout/Navigation:** [Dashboard layouts, mobile-responsiveness requirements, etc.]

## 5. Outstanding Queries & Assumptions
* [List anything that requires clarification or features that were left open-ended]"""

RAG_SYSTEM_PROMPT = """You are a precise technical assistant for a software project.
You have been given relevant excerpts from a structured Product Requirements Document (PRD).
Answer the user's question using ONLY the information present in the provided context chunks.
Be concise, factual, and technically precise.
If the answer cannot be determined from the context, say so explicitly — do not hallucinate."""


def transform_transcript(raw_transcript: str) -> str:
    """Module A: raw transcript → structured Markdown PRD."""
    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Below is the raw meeting transcript. "
                    "Transform it into a structured Markdown PRD per your instructions.\n\n"
                    f"{raw_transcript}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response")
    return content.strip()


def generate_rag_answer(query: str, context_chunks: list[str]) -> str:
    """Module C step 4: retrieved chunks + query → grounded factual answer."""
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"### Context Chunks\n{context_block}\n\n"
                    f"### Question\n{query}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response")
    return content.strip()
