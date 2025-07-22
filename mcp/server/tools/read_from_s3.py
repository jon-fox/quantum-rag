"""Tool for reading transcript files from S3."""

from typing import Dict, Any, Union, List
import boto3
import json

from pydantic import Field, BaseModel, ConfigDict

from ..interfaces.tool import Tool, BaseToolInput, ToolResponse


class ReadFromS3Input(BaseToolInput):
    """Input schema for the ReadFromS3 tool."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"bucket_name": "72b3736a-8a5f-4164-84b4-06121c5a70eb"}]
        }
    )

    bucket_name: str = Field(
        description="The S3 bucket name to read from",
        examples=["72b3736a-8a5f-4164-84b4-06121c5a70eb"],
    )


class ReadFromS3Output(BaseModel):
    """Output schema for the ReadFromS3 tool."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"transcripts": [], "error": None}]}
    )

    transcripts: List[Dict[str, Any]] = Field(
        description="Array of transcript data organized by show"
    )
    error: Union[str, None] = Field(
        default=None, description="An error message if the operation failed."
    )


class ReadFromS3Tool(Tool):
    """Tool that reads transcript files from S3 bucket recursively."""

    name = "ReadFromS3"
    description = (
        "Recursively reads transcript JSON files from S3 bucket organized by shows"
    )
    input_model = ReadFromS3Input
    output_model = ReadFromS3Output

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input": self.input_model.model_json_schema(),
            "output": self.output_model.model_json_schema(),
        }

    async def execute(self, input_data: ReadFromS3Input) -> ToolResponse:
        """Execute the read from S3 tool.

        Args:
            input_data: The validated input for the tool

        Returns:
            A response containing the transcript data
        """
        try:
            s3_client = boto3.client("s3")
            transcripts = []

            # List all objects in the bucket
            paginator = s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=input_data.bucket_name):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]

                    # Check if it's a transcript JSON file
                    if key.endswith(".json") and "transcript" in key:
                        # Parse the path structure: show/episode_id/transcript_file.json
                        path_parts = key.split("/")
                        if len(path_parts) >= 3:
                            show_name = path_parts[0]
                            episode_id = path_parts[1]

                            # Read the JSON file
                            response = s3_client.get_object(
                                Bucket=input_data.bucket_name, Key=key
                            )
                            content = response["Body"].read().decode("utf-8")
                            transcript_data = json.loads(content)

                            # Add metadata to the transcript
                            transcript_entry = {
                                "show_name": show_name,
                                "episode_id": episode_id,
                                "file_path": key,
                                "data": transcript_data,
                            }
                            transcripts.append(transcript_entry)

            output = ReadFromS3Output(transcripts=transcripts, error=None)
            return ToolResponse.from_model(output)

        except Exception as e:
            output = ReadFromS3Output(transcripts=[], error=str(e))
            return ToolResponse.from_model(output)
