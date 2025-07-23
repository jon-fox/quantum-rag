"""Tool for reading transcript files from S3."""

from typing import Dict, Any, Union, List
import boto3
import json
import logging

from pydantic import Field, BaseModel, ConfigDict

from server.interfaces.tool import Tool, BaseToolInput, ToolResponse

logger = logging.getLogger(__name__)


class ReadFromS3Input(BaseToolInput):
    """Input schema for the ReadFromS3 tool."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"show_name": "joe-rogan-experience"}, {}]
        }
    )

    show_name: Union[str, None] = Field(
        default=None,
        description="Optional show name to filter transcripts. If provided, only transcripts from this show will be returned.",
        examples=["joe-rogan-experience"],
    )


class ReadFromS3Output(BaseModel):
    """Output schema for the ReadFromS3 tool."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"transcripts": [], "available_shows": ["joe-rogan-experience", "lex-fridman-podcast"], "error": None}]}
    )

    transcripts: List[Dict[str, Any]] = Field(
        description="Array of transcript data organized by show"
    )
    available_shows: Union[List[str], None] = Field(
        default=None, description="List of available show names when no exact match is found or no show name is provided"
    )
    error: Union[str, None] = Field(
        default=None, description="An error message if the operation failed."
    )


class ReadFromS3Tool(Tool):
    """Tool that reads transcript files from S3 bucket recursively."""

    name = "ReadFromS3"
    description = (
        "Recursively reads transcript JSON files from S3 bucket organized by shows. "
        "The bucket name is retrieved from Parameter Store at /app/app_storage_bucket. "
        "Optionally filter by show name to get transcripts for a specific show. "
        "If no show name is provided or no exact match is found, returns available show names."
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
            logger.info(f"Starting ReadFromS3 execution with show_name: {input_data.show_name}")
            
            ssm_client = boto3.client("ssm", region_name="us-east-1")
            logger.debug("Created SSM client")
            
            try:
                parameter_response = ssm_client.get_parameter(
                    Name="/app/app_storage_bucket"
                )
                bucket_name = parameter_response["Parameter"]["Value"]
                logger.info(f"Retrieved bucket name from SSM: {bucket_name}")
            except ssm_client.exceptions.ParameterNotFound:
                error_msg = "SSM parameter '/app/app_storage_bucket' not found"
                logger.error(error_msg)
                output = ReadFromS3Output(transcripts=[], available_shows=None, error=error_msg)
                return ToolResponse.from_model(output)
            
            s3_client = boto3.client("s3", region_name="us-east-1")
            logger.debug("Created S3 client")
            
            transcripts = []
            available_shows = set()

            paginator = s3_client.get_paginator("list_objects_v2")
            
            # First, check what shows are available
            logger.info("Scanning for available shows")
            for page in paginator.paginate(Bucket=bucket_name):
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".json") and "transcript" in key:
                        path_parts = key.split("/")
                        if len(path_parts) >= 3:
                            available_shows.add(path_parts[0])

            shows_list = sorted(list(available_shows))
            logger.info(f"Found {len(shows_list)} available shows: {shows_list}")
            
            # If a specific show was requested, check if it exists
            if input_data.show_name:
                if input_data.show_name not in available_shows:
                    logger.warning(f"Requested show '{input_data.show_name}' not found")
                    output = ReadFromS3Output(
                        transcripts=[], 
                        available_shows=shows_list, 
                        error=f"Show '{input_data.show_name}' not found. Available shows: {', '.join(shows_list)}"
                    )
                    return ToolResponse.from_model(output)
                
                logger.info(f"Show '{input_data.show_name}' found, loading transcripts")
                
                # Load transcripts for the requested show
                paginate_kwargs = {"Bucket": bucket_name, "Prefix": f"{input_data.show_name}/"}
                for page in paginator.paginate(**paginate_kwargs):
                    if "Contents" not in page:
                        continue
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.endswith(".json") and "transcript" in key:
                            path_parts = key.split("/")
                            if len(path_parts) >= 3:
                                show_name = path_parts[0]
                                episode_id = path_parts[1]
                                
                                try:
                                    response = s3_client.get_object(Bucket=bucket_name, Key=key)
                                    content = response["Body"].read().decode("utf-8")
                                    transcript_data = json.loads(content)

                                    transcript_entry = {
                                        "show_name": show_name,
                                        "episode_id": episode_id,
                                        "file_path": key,
                                        "data": transcript_data,
                                    }
                                    transcripts.append(transcript_entry)
                                    logger.debug(f"Loaded transcript: {key}")
                                except Exception as e:
                                    logger.warning(f"Failed to load {key}: {str(e)}")
                                    continue

            logger.info(f"Execution completed. Found {len(transcripts)} transcripts for {len(shows_list)} available shows")
            output = ReadFromS3Output(
                transcripts=transcripts,
                available_shows=shows_list if not input_data.show_name or len(transcripts) == 0 else None,
                error=None
            )
            return ToolResponse.from_model(output)

        except Exception as e:
            error_msg = f"Error executing ReadFromS3: {str(e)}"
            logger.error(error_msg, exc_info=True)
            output = ReadFromS3Output(transcripts=[], available_shows=None, error=error_msg)
            return ToolResponse.from_model(output)
