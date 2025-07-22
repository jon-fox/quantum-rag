"""Tool for storing embeddings in FAISS index."""

from typing import Dict, Any, Union, List, Optional
import numpy as np
import faiss
import os
import pickle

from pydantic import Field, BaseModel, ConfigDict

from server.interfaces.tool import Tool, BaseToolInput, ToolResponse


class StoreInFaissInput(BaseToolInput):
    """Input schema for the StoreInFaiss tool."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "embeddings": [[0.1, 0.2, 0.3]],
                    "metadata": ["text1"],
                    "index_path": "index.faiss",
                }
            ]
        }
    )

    embeddings: List[List[float]] = Field(
        description="List of embedding vectors to store", examples=[[[0.1, 0.2, 0.3]]]
    )
    metadata: Optional[List[str]] = Field(
        default=None,
        description="Optional metadata associated with each embedding",
        examples=[["text1", "text2"]],
    )
    index_path: str = Field(
        description="Path where the FAISS index will be saved", examples=["index.faiss"]
    )
    dimension: Optional[int] = Field(
        default=None,
        description="Dimension of embeddings (auto-detected if not provided)",
    )


class StoreInFaissOutput(BaseModel):
    """Output schema for the StoreInFaiss tool."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"stored_count": 100, "index_size": 150, "error": None}]
        }
    )

    stored_count: int = Field(description="Number of embeddings stored")
    index_size: int = Field(
        description="Total number of vectors in the index after storage"
    )
    error: Union[str, None] = Field(
        default=None, description="An error message if the operation failed."
    )


class StoreInFaissTool(Tool):
    """Tool that stores embeddings in a FAISS index."""

    name = "StoreInFaiss"
    description = (
        "Stores embedding vectors in a FAISS index for efficient similarity search"
    )
    input_model = StoreInFaissInput
    output_model = StoreInFaissOutput

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input": self.input_model.model_json_schema(),
            "output": self.output_model.model_json_schema(),
        }

    async def execute(self, input_data: StoreInFaissInput) -> ToolResponse:
        """Execute the store in FAISS tool.

        Args:
            input_data: The validated input for the tool

        Returns:
            A response containing storage results
        """
        try:
            if not input_data.embeddings:
                output = StoreInFaissOutput(
                    stored_count=0, index_size=0, error="No embeddings provided"
                )
                return ToolResponse.from_model(output)

            # Convert embeddings to numpy array
            embeddings_array = np.array(input_data.embeddings, dtype=np.float32)
            dimension = input_data.dimension or embeddings_array.shape[1]

            # Load existing index or create new one
            if os.path.exists(input_data.index_path):
                index = faiss.read_index(input_data.index_path)
            else:
                # Create new FAISS index (using IndexFlatL2 for simplicity)
                index = faiss.IndexFlatL2(dimension)

            # Add embeddings to index
            index.add(embeddings_array)

            # Save the updated index
            faiss.write_index(index, input_data.index_path)

            # Save metadata if provided
            if input_data.metadata:
                metadata_path = input_data.index_path.replace(".faiss", "_metadata.pkl")
                existing_metadata = []

                if os.path.exists(metadata_path):
                    with open(metadata_path, "rb") as f:
                        existing_metadata = pickle.load(f)

                existing_metadata.extend(input_data.metadata)

                with open(metadata_path, "wb") as f:
                    pickle.dump(existing_metadata, f)

            output = StoreInFaissOutput(
                stored_count=len(input_data.embeddings),
                index_size=index.ntotal,
                error=None,
            )
            return ToolResponse.from_model(output)

        except Exception as e:
            output = StoreInFaissOutput(stored_count=0, index_size=0, error=str(e))
            return ToolResponse.from_model(output)
