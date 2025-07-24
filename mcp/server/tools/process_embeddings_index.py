"""Tool for creating embeddings index from S3 transcripts in one operation."""

from typing import Dict, Any, Union, List, Optional
import logging

from pydantic import Field, BaseModel, ConfigDict

from server.interfaces.tool import Tool, BaseToolInput, ToolResponse
from .read_from_s3 import ReadFromS3Tool, ReadFromS3Input
from .fetch_embeddings import FetchEmbeddingsTool, FetchEmbeddingsInput
from .store_in_faiss import StoreInFaissTool, StoreInFaissInput

logger = logging.getLogger(__name__)


class ProcessTranscriptsToEmbeddingsInput(BaseToolInput):
    """Input schema for the ProcessTranscriptsToEmbeddings tool."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "show_name": "piers-morgan-uncensored",
                    "index_path": "piers_morgan_index",
                    "embedding_model": "text-embedding-3-small"
                }
            ]
        }
    )

    show_name: str = Field(
        description="Show name to retrieve transcripts for and create embeddings index",
        examples=["piers-morgan-uncensored", "joe-rogan-experience"]
    )
    index_path: str = Field(
        description="Path where the FAISS index will be saved",
        examples=["piers_morgan_index", "joe_rogan_index"]
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use",
        examples=["text-embedding-3-small", "text-embedding-3-large"]
    )
    max_transcripts: Optional[int] = Field(
        default=None,
        description="Maximum number of transcripts to process (for testing/limiting)",
        examples=[5, 10]
    )


class ProcessTranscriptsToEmbeddingsOutput(BaseModel):
    """Output schema for the ProcessTranscriptsToEmbeddings tool."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "success": True,
                    "transcripts_processed": 5,
                    "embeddings_generated": 12,
                    "index_path": "piers_morgan_index.faiss",
                    "metadata_path": "piers_morgan_index_metadata.pkl",
                    "show_name": "piers-morgan-uncensored",
                    "available_shows": None,
                    "error": None
                }
            ]
        }
    )

    success: bool = Field(description="Whether the operation completed successfully")
    transcripts_processed: int = Field(description="Number of transcripts processed")
    embeddings_generated: int = Field(description="Number of embedding vectors generated")
    index_path: str = Field(description="Path to the created FAISS index file")
    metadata_path: str = Field(description="Path to the metadata file")
    show_name: str = Field(description="Show name that was processed")
    available_shows: Union[List[str], None] = Field(default=None, description="List of available shows when the requested show is not found")
    error: Union[str, None] = Field(default=None, description="Error message if operation failed")


class ProcessTranscriptsToEmbeddingsTool(Tool):
    """Tool that processes transcripts from S3 to embeddings stored in FAISS - complete end-to-end workflow."""

    name = "ProcessTranscriptsToEmbeddings"
    description = (
        "Complete workflow that processes transcripts from S3 to embeddings stored in FAISS. "
        "Reads transcripts for a specified show, generates embeddings, and stores them in a FAISS index. "
        "Handles text chunking and embedding generation automatically. "
        "If the requested show is not found, returns available shows for reference."
    )
    input_model = ProcessTranscriptsToEmbeddingsInput
    output_model = ProcessTranscriptsToEmbeddingsOutput

    def __init__(self):
        self.read_from_s3_tool = ReadFromS3Tool()
        self.fetch_embeddings_tool = FetchEmbeddingsTool()
        self.store_in_faiss_tool = StoreInFaissTool()

    def _create_error_output(self, transcripts_count: int = 0, embeddings_count: int = 0, 
                           show_name: str = "", available_shows: Optional[List[str]] = None, 
                           error: str = "") -> ToolResponse:
        """Create standardized error output."""
        return ToolResponse.from_model(ProcessTranscriptsToEmbeddingsOutput(
            success=False,
            transcripts_processed=transcripts_count,
            embeddings_generated=embeddings_count,
            index_path="",
            metadata_path="",
            show_name=show_name,
            available_shows=available_shows,
            error=error
        ))

    async def _read_transcripts_from_s3(self, show_name: str, max_transcripts: Optional[int] = None) -> tuple[List[Dict[str, Any]], Optional[List[str]]]:
        """Read transcripts from S3 for the specified show using the ReadFromS3Tool."""
        read_input = ReadFromS3Input(show_name=show_name)
        response = await self.read_from_s3_tool.execute(read_input)
        
        # Extract data directly from response
        data = response.content[0].json_data if response.content and response.content[0].json_data else {}
        transcripts = data.get('transcripts', [])
        available_shows = data.get('available_shows')
        error = data.get('error')
        
        if error:
            return [], available_shows
            
        if transcripts:
            logger.debug(f"Found {len(transcripts)} transcripts")
        
        if max_transcripts and len(transcripts) > max_transcripts:
            transcripts = transcripts[:max_transcripts]
            logger.info(f"Limited transcripts to {max_transcripts} for processing")
        
        return transcripts, available_shows

    def _extract_texts(self, transcripts: List[Dict[str, Any]]) -> List[str]:
        """Extract text content from transcript data."""
        texts = []
        for transcript in transcripts:
            if not isinstance(transcript, dict):
                continue
                
            transcript_data = transcript.get('data', {})
            text_content = None
            
            # Handle common data structures
            if isinstance(transcript_data, str):
                text_content = transcript_data
            elif isinstance(transcript_data, dict):
                text_content = (
                    transcript_data.get('text') or 
                    transcript_data.get('transcript') or 
                    transcript_data.get('content')
                )
            elif isinstance(transcript_data, list):
                # Join string items, skip non-strings
                text_parts = [item for item in transcript_data if isinstance(item, str)]
                text_content = ' '.join(text_parts) if text_parts else None
            
            if text_content and text_content.strip():
                texts.append(text_content)
        
        logger.info(f"Extracted {len(texts)} text chunks from {len(transcripts)} transcripts")
        return texts

    async def _generate_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings for texts using the FetchEmbeddingsTool."""
        if not texts:
            return []
        
        fetch_input = FetchEmbeddingsInput(texts=texts, model=model)
        response = await self.fetch_embeddings_tool.execute(fetch_input)
        
        # Extract data directly from response
        data = response.content[0].json_data if response.content and response.content[0].json_data else {}
        embeddings = data.get('embeddings', [])
        error = data.get('error')
        
        if error:
            logger.error(f"FetchEmbeddingsTool returned error: {error}")
            return []
            
        logger.info(f"Generated {len(embeddings)} embeddings using FetchEmbeddingsTool")
        return embeddings

    async def _create_faiss_index(self, embeddings: List[List[float]], index_path: str, metadata: Optional[List[str]] = None) -> tuple[str, str]:
        """Create and save FAISS index with embeddings using StoreInFaissTool."""
        if not embeddings:
            raise ValueError("No embeddings provided for index creation")
        
        store_input = StoreInFaissInput(
            embeddings=embeddings,
            index_path=f"{index_path}.faiss",
            metadata=metadata
        )
        
        response = await self.store_in_faiss_tool.execute(store_input)
        
        # Extract data directly from response
        data = response.content[0].json_data if response.content and response.content[0].json_data else {}
        error = data.get('error')
        
        if error:
            raise ValueError(f"Failed to create FAISS index: {error}")
            
        stored_count = data.get('stored_count', 0)
        index_file_path = f"{index_path}.faiss"
        metadata_file_path = f"{index_path}_metadata.pkl"
        
        logger.info(f"Created FAISS index with {stored_count} vectors using StoreInFaissTool")
        return index_file_path, metadata_file_path

    def _find_exact_show_match(self, requested_show: str, available_shows: List[str]) -> Optional[str]:
        """Find exact match for show name (case-insensitive)."""
        if not available_shows:
            return None
        
        requested_lower = requested_show.lower()
        for show in available_shows:
            if show.lower() == requested_lower:
                return show
        return None

    async def execute(self, input_data: ProcessTranscriptsToEmbeddingsInput) -> ToolResponse:
        """Execute the complete embeddings index creation workflow."""
        available_shows = None  # Initialize to preserve for error handling
        
        try:
            logger.info(f"Starting embeddings index creation for show: {input_data.show_name}")
            
            transcripts, available_shows = await self._read_transcripts_from_s3(input_data.show_name, input_data.max_transcripts)
            
            if not transcripts and available_shows:
                exact_match = self._find_exact_show_match(input_data.show_name, available_shows)
                if exact_match and exact_match != input_data.show_name:
                    logger.info(f"Found exact match with different casing: '{exact_match}'")
                    retry_transcripts, _ = await self._read_transcripts_from_s3(exact_match, input_data.max_transcripts)
                    if retry_transcripts:
                        transcripts = retry_transcripts
                        input_data.show_name = exact_match
            
            if not transcripts:
                return self._create_error_output(
                    show_name=input_data.show_name,
                    available_shows=available_shows,
                    error=f"No transcripts found for show '{input_data.show_name}'. Available shows: {', '.join(available_shows) if available_shows else 'None'}"
                )
            
            texts = self._extract_texts(transcripts)
            if not texts:
                return self._create_error_output(
                    transcripts_count=len(transcripts),
                    show_name=input_data.show_name,
                    available_shows=available_shows,
                    error="No text content found in transcripts"
                )
            
            embeddings = await self._generate_embeddings(texts, input_data.embedding_model)
            if not embeddings:
                return self._create_error_output(
                    transcripts_count=len(transcripts),
                    show_name=input_data.show_name,
                    available_shows=available_shows,
                    error="Failed to generate embeddings"
                )
            
            # Create metadata for each transcript
            metadata = [
                f"{transcript.get('show_name', 'unknown')}/{transcript.get('episode_id', 'unknown')}"
                for transcript in transcripts
                if isinstance(transcript, dict)
            ]
            
            index_file_path, metadata_file_path = await self._create_faiss_index(
                embeddings, 
                input_data.index_path,
                metadata
            )
            
            output = ProcessTranscriptsToEmbeddingsOutput(
                success=True,
                transcripts_processed=len(transcripts),
                embeddings_generated=len(embeddings),
                index_path=index_file_path,
                metadata_path=metadata_file_path,
                show_name=input_data.show_name,
                available_shows=None,
                error=None
            )
            
            logger.info(f"Successfully created embeddings index: {len(transcripts)} transcripts → {len(embeddings)} embeddings")
            return ToolResponse.from_model(output)
            
        except Exception as e:
            logger.error(f"Failed to create embeddings index: {e}")
            return self._create_error_output(
                show_name=input_data.show_name,
                available_shows=available_shows,
                error=str(e)
            )
