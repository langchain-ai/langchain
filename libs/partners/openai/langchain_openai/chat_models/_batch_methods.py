"""Batch API methods for BaseChatOpenAI class."""

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


def batch_create(
    self,
    messages_list: List[List[BaseMessage]],
    *,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    poll_interval: float = 10.0,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> str:
    """
    Create a batch job using OpenAI's Batch API for asynchronous processing.
    
    This method provides 50% cost savings compared to the standard API in exchange
    for asynchronous processing with polling for results.
    
    Args:
        messages_list: List of message sequences to process in batch.
        description: Optional description for the batch job.
        metadata: Optional metadata to attach to the batch job.
        poll_interval: Default time in seconds between status checks when polling.
        timeout: Default maximum time in seconds to wait for completion.
        **kwargs: Additional parameters to pass to chat completions.
        
    Returns:
        The batch ID for tracking the asynchronous job.
        
    Raises:
        BatchError: If batch creation fails.
        
    Example:
        .. code-block:: python
        
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            
            llm = ChatOpenAI()
            messages_list = [
                [HumanMessage(content="What is 2+2?")],
                [HumanMessage(content="What is the capital of France?")],
            ]
            
            # Create batch job (50% cost savings)
            batch_id = llm.batch_create(messages_list)
            
            # Later, retrieve results
            results = llm.batch_retrieve(batch_id)
    """
    # Import here to avoid circular imports
    from langchain_openai.chat_models.batch import OpenAIBatchProcessor
    
    # Create batch processor with current model settings
    processor = OpenAIBatchProcessor(
        client=self.root_client,
        model=self.model_name,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    
    # Filter and prepare kwargs for batch processing
    batch_kwargs = self._get_invocation_params(**kwargs)
    # Remove model from kwargs since it's handled by the processor
    batch_kwargs.pop("model", None)
    
    return processor.create_batch(
        messages_list=messages_list,
        description=description,
        metadata=metadata,
        **batch_kwargs,
    )


def batch_retrieve(
    self,
    batch_id: str,
    *,
    poll_interval: Optional[float] = None,
    timeout: Optional[float] = None,
) -> List[ChatResult]:
    """
    Retrieve results from a batch job, polling until completion if necessary.
    
    This method will poll the batch status until completion and return the results
    converted to LangChain ChatResult format.
    
    Args:
        batch_id: The batch ID returned from batch_create().
        poll_interval: Time in seconds between status checks. Uses default if None.
        timeout: Maximum time in seconds to wait. Uses default if None.
        
    Returns:
        List of ChatResult objects corresponding to the original message sequences.
        
    Raises:
        BatchError: If batch retrieval fails, times out, or batch job failed.
        
    Example:
        .. code-block:: python
        
            # After creating a batch job
            batch_id = llm.batch_create(messages_list)
            
            # Retrieve results (will poll until completion)
            results = llm.batch_retrieve(batch_id)
            
            for result in results:
                print(result.generations[0].message.content)
    """
    # Import here to avoid circular imports
    from langchain_openai.chat_models.batch import OpenAIBatchProcessor
    
    # Create batch processor with current model settings
    processor = OpenAIBatchProcessor(
        client=self.root_client,
        model=self.model_name,
        poll_interval=poll_interval or 10.0,
        timeout=timeout,
    )
    
    # Poll for completion and retrieve results
    processor.poll_batch_status(
        batch_id=batch_id,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    
    return processor.retrieve_batch_results(batch_id)
