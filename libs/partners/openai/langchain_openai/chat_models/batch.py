"""OpenAI Batch API client wrapper for LangChain integration."""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import openai
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_openai.chat_models._compat import convert_dict_to_message, convert_message_to_dict


class BatchStatus(str, Enum):
    """OpenAI Batch API status values."""
    
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class BatchError(Exception):
    """Exception raised when batch processing fails."""
    
    def __init__(self, message: str, batch_id: Optional[str] = None, status: Optional[str] = None):
        super().__init__(message)
        self.batch_id = batch_id
        self.status = status


class OpenAIBatchClient:
    """
    OpenAI Batch API client wrapper that handles batch creation, status polling, 
    and result retrieval.
    
    This class provides a high-level interface to OpenAI's Batch API, which offers
    50% cost savings compared to the standard API in exchange for asynchronous processing.
    """
    
    def __init__(self, client: openai.OpenAI):
        """
        Initialize the batch client.
        
        Args:
            client: OpenAI client instance to use for API calls.
        """
        self.client = client
    
    def create_batch(
        self,
        requests: List[Dict[str, Any]],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new batch job with the OpenAI Batch API.
        
        Args:
            requests: List of request objects in OpenAI batch format.
            description: Optional description for the batch job.
            metadata: Optional metadata to attach to the batch job.
            
        Returns:
            The batch ID for tracking the job.
            
        Raises:
            BatchError: If batch creation fails.
        """
        try:
            # Create JSONL content for the batch
            jsonl_content = "\n".join(json.dumps(req) for req in requests)
            
            # Upload the batch file
            file_response = self.client.files.create(
                file=jsonl_content.encode('utf-8'),
                purpose="batch"
            )
            
            # Create the batch job
            batch_response = self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata or {}
            )
            
            return batch_response.id
            
        except openai.OpenAIError as e:
            raise BatchError(f"Failed to create batch: {e}") from e
        except Exception as e:
            raise BatchError(f"Unexpected error creating batch: {e}") from e
    
    def retrieve_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieve batch information by ID.
        
        Args:
            batch_id: The batch ID to retrieve.
            
        Returns:
            Dictionary containing batch information including status.
            
        Raises:
            BatchError: If batch retrieval fails.
        """
        try:
            batch = self.client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": getattr(batch, 'completed_at', None),
                "failed_at": getattr(batch, 'failed_at', None),
                "expired_at": getattr(batch, 'expired_at', None),
                "request_counts": getattr(batch, 'request_counts', {}),
                "metadata": getattr(batch, 'metadata', {}),
                "errors": getattr(batch, 'errors', None),
                "output_file_id": getattr(batch, 'output_file_id', None),
                "error_file_id": getattr(batch, 'error_file_id', None),
            }
        except openai.OpenAIError as e:
            raise BatchError(f"Failed to retrieve batch {batch_id}: {e}", batch_id=batch_id) from e
        except Exception as e:
            raise BatchError(f"Unexpected error retrieving batch {batch_id}: {e}", batch_id=batch_id) from e
    
    def poll_batch_status(
        self,
        batch_id: str,
        poll_interval: float = 10.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Poll batch status until completion or failure.
        
        Args:
            batch_id: The batch ID to poll.
            poll_interval: Time in seconds between status checks.
            timeout: Maximum time in seconds to wait. None for no timeout.
            
        Returns:
            Final batch information when completed.
            
        Raises:
            BatchError: If batch fails or times out.
        """
        start_time = time.time()
        
        while True:
            batch_info = self.retrieve_batch(batch_id)
            status = batch_info["status"]
            
            if status == BatchStatus.COMPLETED:
                return batch_info
            elif status in [BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
                error_msg = f"Batch {batch_id} failed with status: {status}"
                if batch_info.get("errors"):
                    error_msg += f". Errors: {batch_info['errors']}"
                raise BatchError(error_msg, batch_id=batch_id, status=status)
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise BatchError(
                    f"Batch {batch_id} timed out after {timeout} seconds. Current status: {status}",
                    batch_id=batch_id,
                    status=status
                )
            
            time.sleep(poll_interval)
    
    def retrieve_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve results from a completed batch.
        
        Args:
            batch_id: The batch ID to retrieve results for.
            
        Returns:
            List of result objects from the batch.
            
        Raises:
            BatchError: If batch is not completed or result retrieval fails.
        """
        try:
            batch_info = self.retrieve_batch(batch_id)
            
            if batch_info["status"] != BatchStatus.COMPLETED:
                raise BatchError(
                    f"Batch {batch_id} is not completed. Current status: {batch_info['status']}",
                    batch_id=batch_id,
                    status=batch_info["status"]
                )
            
            output_file_id = batch_info.get("output_file_id")
            if not output_file_id:
                raise BatchError(f"No output file found for batch {batch_id}", batch_id=batch_id)
            
            # Download and parse the results file
            file_content = self.client.files.content(output_file_id)
            results = []
            
            for line in file_content.text.strip().split('\n'):
                if line.strip():
                    results.append(json.loads(line))
            
            return results
            
        except openai.OpenAIError as e:
            raise BatchError(f"Failed to retrieve results for batch {batch_id}: {e}", batch_id=batch_id) from e
        except Exception as e:
            raise BatchError(f"Unexpected error retrieving results for batch {batch_id}: {e}", batch_id=batch_id) from e
    
    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Cancel a batch job.
        
        Args:
            batch_id: The batch ID to cancel.
            
        Returns:
            Updated batch information after cancellation.
            
        Raises:
            BatchError: If batch cancellation fails.
        """
        try:
            batch = self.client.batches.cancel(batch_id)
            return self.retrieve_batch(batch_id)
        except openai.OpenAIError as e:
            raise BatchError(f"Failed to cancel batch {batch_id}: {e}", batch_id=batch_id) from e
        except Exception as e:
            raise BatchError(f"Unexpected error cancelling batch {batch_id}: {e}", batch_id=batch_id) from e


def create_batch_request(
    messages: List[BaseMessage],
    model: str,
    custom_id: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Create a batch request object from LangChain messages.
    
    Args:
        messages: List of LangChain messages to convert.
        model: The model to use for the request.
        custom_id: Unique identifier for this request within the batch.
        **kwargs: Additional parameters to pass to the chat completion.
        
    Returns:
        Dictionary in OpenAI batch request format.
    """
    # Convert LangChain messages to OpenAI format
    openai_messages = [convert_message_to_dict(msg) for msg in messages]
    
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": openai_messages,
            **kwargs
        }
    }

