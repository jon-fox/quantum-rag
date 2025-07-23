"""Tool for fetching embeddings using OpenAI."""

from typing import Dict, Any, Union, List
import boto3
import openai
import logging

from pydantic import Field, BaseModel, ConfigDict

from server.interfaces.tool import Tool, BaseToolInput, ToolResponse

logger = logging.getLogger(__name__)


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
        "The OpenAI API key is retrieved from Parameter Store at /openai/api_key. "
        "Automatically chunks long texts to avoid token limits."
    )
    input_model = FetchEmbeddingsInput
    output_model = FetchEmbeddingsOutput

    def chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Chunk text into smaller pieces to avoid token limits.
        
        Args:
            text: The text to chunk
            max_tokens: Maximum tokens per chunk (conservative estimate)
            
        Returns:
            List of text chunks
        """
        # Rough estimate: 1 token ≈ 4 characters for English text
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # If we're not at the end, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence endings within the last 500 chars
                break_point = text.rfind('.', start + max_chars - 500, end)
                if break_point == -1:
                    break_point = text.rfind('\n', start + max_chars - 500, end)
                if break_point == -1:
                    break_point = text.rfind(' ', start + max_chars - 500, end)
                
                if break_point > start:
                    end = break_point + 1
            
            chunks.append(text[start:end].strip())
            start = end
        
        return chunks

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
            logger.info(f"Fetching embeddings for {len(input_data.texts)} text(s)")
            
            ssm_client = boto3.client("ssm", region_name="us-east-1")
            parameter_response = ssm_client.get_parameter(
                Name="/openai/api_key",
                WithDecryption=True
            )
            api_key = parameter_response["Parameter"]["Value"]

            client = openai.OpenAI(api_key=api_key)
            
            # Process each text and chunk if necessary
            all_embeddings = []
            
            for text in input_data.texts:
                chunks = self.chunk_text(text)
                logger.info(f"Text chunked into {len(chunks)} pieces")
                
                # Get embeddings for each chunk
                for chunk in chunks:
                    try:
                        response = client.embeddings.create(
                            model=input_data.model, 
                            input=[chunk]
                        )
                        embeddings = [embedding.embedding for embedding in response.data]
                        all_embeddings.extend(embeddings)
                        logger.debug(f"Generated embedding for chunk of length {len(chunk)}")
                    except Exception as chunk_error:
                        logger.warning(f"Failed to generate embedding for chunk: {str(chunk_error)}")
                        continue

            logger.info(f"Generated {len(all_embeddings)} total embeddings")
            output = FetchEmbeddingsOutput(embeddings=all_embeddings, error=None)
            return ToolResponse.from_model(output)

        except Exception as e:
            error_msg = f"Error fetching embeddings: {str(e)}"
            logger.error(error_msg)
            output = FetchEmbeddingsOutput(embeddings=[], error=error_msg)
            return ToolResponse.from_model(output)
