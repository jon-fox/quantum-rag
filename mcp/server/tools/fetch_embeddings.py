"""Tool for fetching embeddings using OpenAI."""

from typing import Dict, Any, Union, List
import boto3
import openai

from pydantic import Field, BaseModel, ConfigDict

from server.interfaces.tool import Tool, BaseToolInput, ToolResponse


class FetchEmbeddingsInput(BaseToolInput):
    """Input schema for the FetchEmbeddings tool."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texts": ["Hello world", "This is a test"],
                    "model": "text-embedding-3-small",
                }
            ]
        }
    )

    texts: List[str] = Field(
        description="List of texts to generate embeddings for",
        examples=[["Hello world", "This is a test"]],
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use",
        examples=["text-embedding-3-small", "text-embedding-3-large"],
    )


class FetchEmbeddingsOutput(BaseModel):
    """Output schema for the FetchEmbeddings tool."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"embeddings": [], "error": None}]}
    )

    embeddings: List[List[float]] = Field(
        description="List of embedding vectors for each input text"
    )
    error: Union[str, None] = Field(
        default=None, description="An error message if the operation failed."
    )


class FetchEmbeddingsTool(Tool):
    """Tool that fetches embeddings from OpenAI for given texts."""

    name = "FetchEmbeddings"
    description = (
        "Fetches embeddings from OpenAI for a list of input texts. "
        "The OpenAI API key is retrieved from Parameter Store at /openai/api_key."
    )
    input_model = FetchEmbeddingsInput
    output_model = FetchEmbeddingsOutput

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input": self.input_model.model_json_schema(),
            "output": self.output_model.model_json_schema(),
        }

    async def execute(self, input_data: FetchEmbeddingsInput) -> ToolResponse:
        """Execute the fetch embeddings tool.

        Args:
            input_data: The validated input for the tool

        Returns:
            A response containing the embeddings
        """
        try:
            ssm_client = boto3.client("ssm", region_name="us-east-1")
            parameter_response = ssm_client.get_parameter(
                Name="/openai/api_key",
                WithDecryption=True
            )
            api_key = parameter_response["Parameter"]["Value"]

            client = openai.OpenAI(api_key=api_key)

            response = client.embeddings.create(
                model=input_data.model, input=input_data.texts
            )

            embeddings = [embedding.embedding for embedding in response.data]

            output = FetchEmbeddingsOutput(embeddings=embeddings, error=None)
            return ToolResponse.from_model(output)

        except Exception as e:
            output = FetchEmbeddingsOutput(embeddings=[], error=str(e))
            return ToolResponse.from_model(output)
