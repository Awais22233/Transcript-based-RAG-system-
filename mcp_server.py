"""
Module D: MCP (Model Context Protocol) Server.

Exposes the tool:
    fetch_project_requirements(project_id, search_query)

Allows IDE assistants (Cursor, VS Code + Continue, etc.) to query
project requirements directly while writing code.

Run standalone:
    python mcp_server.py

Configure in Cursor (.cursor/mcp.json):
{
  "mcpServers": {
    "hubmicro-ai-scribe": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "C:/path/to/hubmicro-ai-scribe",
      "env": {
        "GROQ_API_KEY": "your_key_here"
      }
    }
  }
}
"""
import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.services.chroma import query_chunks
from app.services.llm import generate_rag_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hubmicro-mcp")

server = Server("hubmicro-ai-scribe")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="fetch_project_requirements",
            description=(
                "Search the indexed project requirements and return a factual, "
                "concise answer grounded in the stored PRD chunks. "
                "Use this to answer questions like 'What color palette did the client want?' "
                "or 'What authentication method was specified?' or 'What are the performance requirements?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "The unique project identifier to search within.",
                    },
                    "search_query": {
                        "type": "string",
                        "description": "Natural language question about the project requirements.",
                    },
                },
                "required": ["project_id", "search_query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "fetch_project_requirements":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    project_id: str = arguments.get("project_id", "").strip()
    search_query: str = arguments.get("search_query", "").strip()

    if not project_id or not search_query:
        return [
            TextContent(
                type="text",
                text="Error: both project_id and search_query are required.",
            )
        ]

    try:
        chunks = query_chunks(project_id=project_id, query=search_query)

        if not chunks:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"No requirement chunks found for project '{project_id}'. "
                        "Make sure the project transcript has been ingested first via POST /api/v1/ingest."
                    ),
                )
            ]

        answer = generate_rag_answer(search_query, chunks)

        result = {
            "project_id": project_id,
            "query": search_query,
            "answer": answer,
            "chunks_used": len(chunks),
            "source_chunks": chunks,
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

    except Exception as exc:
        logger.exception("Error in fetch_project_requirements")
        return [TextContent(type="text", text=f"Error: {exc}")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
