"""
HTTP Stream transport client for MCP Agent example.
Communicates with the server_http.py `/mcp` endpoint using HTTP GET/POST/DELETE for JSON-RPC streams.
"""

from atomic_agents.connectors.mcp import fetch_mcp_tools, MCPTransportType
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from atomic_agents import BaseIOSchema, BaseAgent, BaseAgentConfig
import sys
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from pydantic import Field
import openai
import os
import boto3
import instructor
from typing import Union, Type, Dict
from dataclasses import dataclass


@dataclass
class MCPConfig:
    """Configuration for the MCP Agent system using HTTP Stream transport."""

    mcp_server_url: str = "http://localhost:6969"
    openai_model: str = "gpt-4o"

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from AWS Parameter Store, fallback to environment variable."""
        try:
            ssm_client = boto3.client("ssm", region_name="us-east-1")
            parameter_response = ssm_client.get_parameter(
                Name="/openai/api_key",
                WithDecryption=True
            )
            return parameter_response["Parameter"]["Value"]
        except Exception:
            # Fallback to environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Could not get OpenAI API key from Parameter Store or OPENAI_API_KEY environment variable")
            return api_key


def main():
    # Use default HTTP transport settings from MCPConfig
    config = MCPConfig()
    console = Console()
    api_key = config.get_openai_api_key()
    client = instructor.from_openai(openai.OpenAI(api_key=api_key))

    console.print("[bold green]Initializing MCP Agent System (HTTP Stream mode)...[/bold green]")
    tools = fetch_mcp_tools(mcp_endpoint=config.mcp_server_url, transport_type=MCPTransportType.HTTP_STREAM)
    if not tools:
        console.print(f"[bold red]No MCP tools found at {config.mcp_server_url}[/bold red]")
        sys.exit(1)

    # Display available tools
    table = Table(title="Available MCP Tools", box=None)
    table.add_column("Tool Name", style="cyan")
    table.add_column("Input Schema", style="yellow")
    table.add_column("Description", style="magenta")
    for ToolClass in tools:
        schema_name = getattr(ToolClass.input_schema, "__name__", "N/A")
        table.add_row(ToolClass.mcp_tool_name, schema_name, ToolClass.__doc__ or "")
    console.print(table)

    # Build orchestrator
    class MCPOrchestratorInputSchema(BaseIOSchema):
        """Input schema for the MCP orchestrator that processes user queries."""

        query: str = Field(...)

    class FinalResponseSchema(BaseIOSchema):
        """Schema for the final response to the user."""

        response_text: str = Field(...)

    # Map schemas and define ActionUnion
    tool_schema_map: Dict[Type[BaseIOSchema], Type] = {
        ToolClass.input_schema: ToolClass for ToolClass in tools if hasattr(ToolClass, "input_schema")
    }
    available_schemas = tuple(tool_schema_map.keys()) + (FinalResponseSchema,)
    ActionUnion = Union[available_schemas]

    class OrchestratorOutputSchema(BaseIOSchema):
        """Output schema for the MCP orchestrator containing reasoning and selected action."""

        reasoning: str
        action: ActionUnion

    history = ChatHistory()
    orchestrator_agent = BaseAgent[MCPOrchestratorInputSchema, OrchestratorOutputSchema](
        BaseAgentConfig(
            client=client,
            model=config.openai_model,
            history=history,
            system_prompt_generator=SystemPromptGenerator(
                background=[
                    "You are an MCP Orchestrator Agent, designed to chat with users and",
                    "determine the best way to handle their queries using the available tools.",
                    "You have access to individual tools (ReadFromS3, FetchEmbeddings, StoreInFaiss) and",
                    "a comprehensive ProcessTranscriptsToEmbeddings tool that handles the complete workflow:",
                    "S3 transcript retrieval → embedding generation → FAISS index creation in one operation."
                ],
                steps=[
                    "1. Use the reasoning field to determine if one or more tool calls could be used to handle the user's query.",
                    "2. For complete workflows (getting transcripts + embeddings + FAISS storage), ALWAYS use ProcessTranscriptsToEmbeddings tool.",
                    "3. For individual operations, use specific tools: ReadFromS3, FetchEmbeddings, or StoreInFaiss.",
                    "4. If a tool fails, analyze the error and suggest alternatives or fixes to the user.",
                    "5. If a tool returns an error with available alternatives (like available_shows), automatically retry with the closest matching option.",
                    "6. If no tool can handle the query, provide a clear explanation and suggestions.",
                    "7. When finished processing, provide a final response to the user."
                ],
                output_instructions=[
                    "1. Always provide a detailed explanation of your decision-making process in the 'reasoning' field.",
                    "2. Choose exactly one action schema (either a tool input or FinalResponseSchema).",
                    "3. Ensure all required parameters for the chosen tool are properly extracted and validated.",
                    "4. IMPORTANT: When users request complete workflows like 'get transcript, create embeddings, store in FAISS' - use ProcessTranscriptsToEmbeddings tool.",
                    "5. For single-step operations, use individual tools (ReadFromS3, FetchEmbeddings, StoreInFaiss).",
                    "6. When tools fail, acknowledge the error and provide helpful suggestions or alternatives.",
                    "7. Look for similar names in available_shows when exact matches fail (e.g., 'The Tim Dillon Show' vs 'The_Tim_Dillon_Show').",
                    "8. Maintain a professional and helpful tone in all responses.",
                    "9. Provide clear final responses when the task is complete or cannot be completed."
                ],
            ),
        )
    )

    console.print("[bold green]HTTP Stream client ready. Type 'exit' to quit.[/bold green]")
    
    while True:
        query = console.input("[bold yellow]You:[/bold yellow] ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        try:
            # Initial run with user query
            orchestrator_output = orchestrator_agent.run(MCPOrchestratorInputSchema(query=query))

            # Debug output to see what's actually in the output
            console.print(
                f"[dim]Debug - orchestrator_output type: {type(orchestrator_output)}, reasoning: {getattr(orchestrator_output, 'reasoning', 'N/A')[:100]}..."
            )

            # Handle the output similar to SSE version
            if hasattr(orchestrator_output, "chat_message") and not hasattr(orchestrator_output, "action"):
                # Convert BaseAgentOutputSchema to FinalResponseSchema
                action_instance = FinalResponseSchema(response_text=orchestrator_output.chat_message)
                reasoning = "Response generated directly from chat model"
            elif hasattr(orchestrator_output, "action"):
                action_instance = orchestrator_output.action
                reasoning = (
                    orchestrator_output.reasoning if hasattr(orchestrator_output, "reasoning") else "No reasoning provided"
                )
            else:
                console.print("[yellow]Warning: Unexpected response format. Unable to process.[/yellow]")
                continue

            console.print(f"[cyan]Orchestrator reasoning:[/cyan] {reasoning}")

            # Keep executing until we get a final response
            while not isinstance(action_instance, FinalResponseSchema):
                # Find the matching tool class
                tool_class = tool_schema_map.get(type(action_instance))
                if not tool_class:
                    console.print(f"[red]Error: No tool found for schema {type(action_instance)}[/red]")
                    action_instance = FinalResponseSchema(
                        response_text="I encountered an internal error. Could not find the appropriate tool."
                    )
                    break

                # Execute the tool
                console.print(f"[blue]Executing {tool_class.mcp_tool_name}...[/blue]")
                # Only show key parameters, not potentially large data
                params_display = {k: v for k, v in action_instance.model_dump().items() 
                                if k not in ['texts', 'embeddings', 'data', 'content'] and not isinstance(v, (list, dict)) or k in ['show_name', 'index_path', 'model']}
                if len(params_display) != len(action_instance.model_dump()):
                    params_display['...'] = 'additional parameters omitted'
                console.print(f"[dim]Parameters: {params_display}")
                
                tool_instance = tool_class()
                try:
                    result = tool_instance.run(action_instance)
                    
                    # Check if the tool result indicates an error
                    if hasattr(result, 'result'):
                        result_data = result.result
                        
                        # Handle structured error responses (e.g., from ProcessTranscriptsToEmbeddings)
                        if hasattr(result_data, 'success') and not result_data.success:
                            error_msg = getattr(result_data, 'error', 'Unknown error occurred')
                            available_shows = getattr(result_data, 'available_shows', None)
                            
                            console.print(f"[red]Tool Error:[/red] {error_msg}")
                            if available_shows:
                                console.print(f"[yellow]Available shows:[/yellow] {', '.join(available_shows)}")
                            
                            # Ask agent to handle the error or provide alternatives
                            error_context = f"The {tool_class.mcp_tool_name} tool failed with error: {error_msg}"
                            if available_shows:
                                error_context += f" Available shows are: {', '.join(available_shows)}"
                            error_context += f". Please provide alternative suggestions or retry with a similar show name for: {query}"
                            
                            error_query = error_context
                            next_output = orchestrator_agent.run(MCPOrchestratorInputSchema(query=error_query))
                            
                            if hasattr(next_output, "action"):
                                action_instance = next_output.action
                                if hasattr(next_output, "reasoning"):
                                    console.print(f"[cyan]Orchestrator reasoning:[/cyan] {next_output.reasoning}")
                            else:
                                action_instance = FinalResponseSchema(response_text=next_output.chat_message)
                            continue
                    
                    # Display successful result
                    if tool_class.mcp_tool_name in ["ReadFromS3", "FetchEmbeddings", "StoreInFaiss", "ProcessTranscriptsToEmbeddings"]:
                        console.print(f"[bold green]Result:[/bold green] {tool_class.mcp_tool_name} completed successfully")
                    else:
                        console.print(f"[bold green]Result:[/bold green] {result.result}")

                    # Ask orchestrator what to do next with the result
                    next_query = f"Based on the tool result: {result.result}, please provide the next step or final response for: {query}"
                    next_output = orchestrator_agent.run(MCPOrchestratorInputSchema(query=next_query))

                    # Debug output for subsequent responses
                    console.print(
                        f"[dim]Debug - subsequent orchestrator_output type: {type(next_output)}, reasoning: {getattr(next_output, 'reasoning', 'N/A')[:100]}..."
                    )

                    if hasattr(next_output, "action"):
                        action_instance = next_output.action
                        if hasattr(next_output, "reasoning"):
                            console.print(f"[cyan]Orchestrator reasoning:[/cyan] {next_output.reasoning}")
                    else:
                        # If no action, treat as final response
                        action_instance = FinalResponseSchema(response_text=next_output.chat_message)

                except Exception as e:
                    console.print(f"[red]Error executing tool: {e}[/red]")
                    action_instance = FinalResponseSchema(
                        response_text=f"I encountered an error while executing the tool: {str(e)}"
                    )
                    break

            # Display final response
            if isinstance(action_instance, FinalResponseSchema):
                md = Markdown(action_instance.response_text)
                console.print("[bold blue]Agent:[/bold blue]")
                console.print(md)
            else:
                console.print("[red]Error: Expected final response but got something else[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
