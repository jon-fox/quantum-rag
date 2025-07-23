"""example-mcp-server MCP Server HTTP Stream Transport."""

from typing import List
import argparse

import uvicorn
from starlette.middleware.cors import CORSMiddleware

from fastmcp import FastMCP

from server.services.tool_service import ToolService
from server.interfaces.tool import Tool
from server.tools import FetchEmbeddingsTool, ReadFromS3Tool, StoreInFaissTool, ProcessTranscriptsToEmbeddingsTool


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    mcp = FastMCP("example-mcp-server")
    tool_service = ToolService()

    tool_service.register_tools(
        [
            FetchEmbeddingsTool(),
            ReadFromS3Tool(),
            StoreInFaissTool(),
            ProcessTranscriptsToEmbeddingsTool(),
        ]
    )
    tool_service.register_mcp_handlers(mcp)

    return mcp


def create_http_app():
    """Create a FastMCP HTTP app with CORS middleware."""
    mcp_server = create_mcp_server()

    # Use FastMCP directly as the app instead of mounting it
    # This avoids the task group initialization issue
    # See: https://github.com/modelcontextprotocol/python-sdk/issues/732
    app = mcp_server.streamable_http_app()  # type: ignore[attr-defined]

    app = CORSMiddleware(
        app,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    return app


def main():
    """Entry point for the HTTP Stream Transport server."""
    parser = argparse.ArgumentParser(description="Run MCP HTTP Stream server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=6969, help="Port to listen on")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    app = create_http_app()
    print(f"MCP HTTP Stream Server starting on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
