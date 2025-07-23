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

    async def _read_transcripts_from_s3(self, show_name: str, max_transcripts: Optional[int] = None) -> tuple[List[Dict[str, Any]], Optional[List[str]]]:
        """Read transcripts from S3 for the specified show using the ReadFromS3Tool."""
        read_input = ReadFromS3Input(show_name=show_name)
        response = await self.read_from_s3_tool.execute(read_input)
        
        # Extract the actual data from the ToolResponse
        if response.content and len(response.content) > 0:
            content = response.content[0]
            if content.type == "json" and content.json_data:
                data = content.json_data
                transcripts = data.get('transcripts', [])
                available_shows = data.get('available_shows', None)
                error = data.get('error', None)
                
                # Debug: Log transcript structure
                if transcripts:
                    logger.debug(f"Found {len(transcripts)} transcripts")
                    logger.debug(f"First transcript type: {type(transcripts[0])}")
                    if isinstance(transcripts[0], dict):
                        logger.debug(f"First transcript keys: {list(transcripts[0].keys())}")
                    else:
                        logger.debug(f"First transcript preview: {str(transcripts[0])[:100]}...")
                
                if error:
                    logger.error(f"ReadFromS3Tool returned error: {error}")
                    return [], available_shows
                
                # Apply max_transcripts limit if specified
                if max_transcripts and len(transcripts) > max_transcripts:
                    transcripts = transcripts[:max_transcripts]
                    logger.info(f"Limited transcripts to {max_transcripts} for processing")
                
                return transcripts, available_shows
        
        # If we get here, something went wrong
        logger.error("Failed to parse response from ReadFromS3Tool")
        return [], None

    def _extract_texts(self, transcripts: List[Dict[str, Any]]) -> List[str]:
        """Extract text content from transcript data."""
        texts = []
        for i, transcript in enumerate(transcripts):
            try:
                # Ensure transcript is a dictionary
                if not isinstance(transcript, dict):
                    logger.warning(f"Transcript {i} is not a dictionary, got {type(transcript)}")
                    continue
                
                transcript_data = transcript.get('data', {})
                
                # Handle different transcript_data types
                if isinstance(transcript_data, dict):
                    # Try different possible text fields
                    text_content = (
                        transcript_data.get('text') or 
                        transcript_data.get('transcript') or 
                        transcript_data.get('content') or
                        f"[Non-text data: {type(transcript_data).__name__}]"
                    )
                elif isinstance(transcript_data, list):
                    # If transcript_data is a list, join all string elements
                    text_parts = []
                    for item in transcript_data:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict):
                            # Extract text from dict items in the list
                            item_text = (
                                item.get('text') or 
                                item.get('transcript') or 
                                item.get('content') or
                                item.get('speaker') or
                                str(item) if len(str(item)) < 200 else f"[Dict: {len(item)} keys]"
                            )
                            if isinstance(item_text, str):
                                text_parts.append(item_text)
                    text_content = ' '.join(text_parts) if text_parts else f"[List with {len(transcript_data)} items]"
                elif isinstance(transcript_data, str):
                    # If transcript_data is already a string, use it directly
                    text_content = transcript_data
                else:
                    text_content = f"[Unknown data type: {type(transcript_data).__name__}]"
                
                if text_content and isinstance(text_content, str) and text_content.strip():
                    texts.append(text_content)
                    
            except Exception as e:
                logger.warning(f"Failed to extract text from transcript {i}: {e}")
                continue
        
        logger.info(f"Extracted {len(texts)} text chunks from {len(transcripts)} transcripts")
        return texts

    async def _generate_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings for texts using the FetchEmbeddingsTool."""
        if not texts:
            return []
        
        fetch_input = FetchEmbeddingsInput(texts=texts, model=model)
        response = await self.fetch_embeddings_tool.execute(fetch_input)
        
        # Extract the actual data from the ToolResponse
        if response.content and len(response.content) > 0:
            content = response.content[0]
            if content.type == "json" and content.json_data:
                data = content.json_data
                embeddings = data.get('embeddings', [])
                error = data.get('error', None)
                
                if error:
                    logger.error(f"FetchEmbeddingsTool returned error: {error}")
                    return []
                
                logger.info(f"Generated {len(embeddings)} embeddings using FetchEmbeddingsTool")
                return embeddings
        
        logger.error("Failed to parse response from FetchEmbeddingsTool")
        return []

    async def _create_faiss_index(self, embeddings: List[List[float]], index_path: str, metadata: Optional[List[str]] = None) -> tuple[str, str]:
        """Create and save FAISS index with embeddings using StoreInFaissTool."""
        if not embeddings:
            raise ValueError("No embeddings provided for index creation")
        
        # Use StoreInFaissTool to create the index
        store_input = StoreInFaissInput(
            embeddings=embeddings,
            index_path=f"{index_path}.faiss",
            metadata=metadata
        )
        
        response = await self.store_in_faiss_tool.execute(store_input)
        
        # Extract the actual data from the ToolResponse
        if response.content and len(response.content) > 0:
            content = response.content[0]
            if content.type == "json" and content.json_data:
                data = content.json_data
                stored_count = data.get('stored_count', 0)
                error = data.get('error', None)
                
                if error:
                    logger.error(f"StoreInFaissTool returned error: {error}")
                    raise ValueError(f"Failed to create FAISS index: {error}")
                
                index_file_path = f"{index_path}.faiss"
                metadata_file_path = f"{index_path}_metadata.pkl"
                
                logger.info(f"Created FAISS index with {stored_count} vectors using StoreInFaissTool")
                return index_file_path, metadata_file_path
        
        logger.error("Failed to parse response from StoreInFaissTool")
        raise ValueError("Failed to create FAISS index: Unknown error from StoreInFaissTool")

    async def _try_find_similar_show(self, requested_show: str, available_shows: List[str]) -> Optional[str]:
        """Try to find a similar show name from available shows."""
        if not available_shows:
            return None
        
        requested_lower = requested_show.lower()
        
        # First, try exact case-insensitive match
        for show in available_shows:
            if show.lower() == requested_lower:
                return show
        
        # Then try partial matching - check if any available show contains key words
        requested_words = set(requested_lower.replace('-', ' ').split())
        
        best_match = None
        best_score = 0
        
        for show in available_shows:
            show_words = set(show.lower().replace('-', ' ').split())
            # Count common words
            common_words = requested_words.intersection(show_words)
            score = len(common_words)
            
            if score > best_score and score > 0:
                best_score = score
                best_match = show
        
        return best_match

    async def execute(self, input_data: ProcessTranscriptsToEmbeddingsInput) -> ToolResponse:
        """Execute the complete embeddings index creation workflow."""
        try:
            logger.info(f"Starting embeddings index creation for show: {input_data.show_name}")
            
            # Step 1: Read transcripts from S3 using ReadFromS3Tool
            transcripts, available_shows = await self._read_transcripts_from_s3(input_data.show_name, input_data.max_transcripts)
            
            # If no transcripts found, try to find a similar show name and retry once
            if not transcripts and available_shows:
                similar_show = await self._try_find_similar_show(input_data.show_name, available_shows)
                if similar_show and similar_show != input_data.show_name:
                    logger.info(f"No exact match for '{input_data.show_name}', trying similar show: '{similar_show}'")
                    transcripts, _ = await self._read_transcripts_from_s3(similar_show, input_data.max_transcripts)
                    if transcripts:
                        # Update the show name for the rest of the process
                        input_data.show_name = similar_show
            
            if not transcripts:
                return ToolResponse.from_model(ProcessTranscriptsToEmbeddingsOutput(
                    success=False,
                    transcripts_processed=0,
                    embeddings_generated=0,
                    index_path="",
                    metadata_path="",
                    show_name=input_data.show_name,
                    available_shows=available_shows,
                    error=f"No transcripts found for show '{input_data.show_name}'. Available shows: {', '.join(available_shows) if available_shows else 'None'}"
                ))
            
            # Step 2: Extract texts
            texts = self._extract_texts(transcripts)
            if not texts:
                return ToolResponse.from_model(ProcessTranscriptsToEmbeddingsOutput(
                    success=False,
                    transcripts_processed=len(transcripts),
                    embeddings_generated=0,
                    index_path="",
                    metadata_path="",
                    show_name=input_data.show_name,
                    available_shows=None,
                    error="No text content found in transcripts"
                ))
            
            # Step 3: Generate embeddings
            embeddings = await self._generate_embeddings(texts, input_data.embedding_model)
            if not embeddings:
                return ToolResponse.from_model(ProcessTranscriptsToEmbeddingsOutput(
                    success=False,
                    transcripts_processed=len(transcripts),
                    embeddings_generated=0,
                    index_path="",
                    metadata_path="",
                    show_name=input_data.show_name,
                    available_shows=None,
                    error="Failed to generate embeddings"
                ))
            
            # Step 4: Create FAISS index
            # Simple metadata - just episode identifiers
            metadata = []
            for transcript in transcripts:
                try:
                    if isinstance(transcript, dict):
                        show_name = transcript.get('show_name', 'unknown')
                        episode_id = transcript.get('episode_id', 'unknown')
                        metadata.append(f"{show_name}/{episode_id}")
                    else:
                        logger.warning(f"Transcript is not a dictionary for metadata: {type(transcript)}")
                        metadata.append("unknown/unknown")
                except Exception as e:
                    logger.warning(f"Failed to create metadata for transcript: {e}")
                    metadata.append("unknown/unknown")
            
            index_file_path, metadata_file_path = await self._create_faiss_index(
                embeddings, 
                input_data.index_path,
                metadata
            )
            
            # Success!
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
            return ToolResponse.from_model(ProcessTranscriptsToEmbeddingsOutput(
                success=False,
                transcripts_processed=0,
                embeddings_generated=0,
                index_path="",
                metadata_path="",
                show_name=input_data.show_name,
                available_shows=None,
                error=str(e)
            ))
