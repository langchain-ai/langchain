#!/usr/bin/env python3
"""Script to add batch() method override to BaseChatOpenAI class."""

import re

def add_batch_override():
    file_path = '/home/daytona/langchain/libs/partners/openai/langchain_openai/chat_models/base.py'
    
    # Read the base.py file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the location to insert the method (after batch_retrieve method)
    pattern = r'(\s+)def _get_generation_chunk_from_completion\('
    match = re.search(pattern, content)
    
    if not match:
        print("Could not find insertion point")
        return False
    
    indent = match.group(1)
    insert_pos = match.start()
    
    # Define the method to insert
    method = f'''
    @override
    def batch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        use_batch_api: bool = False,
        **kwargs: Any,
    ) -> List[BaseMessage]:
        """
        Batch process multiple inputs using either standard API or OpenAI Batch API.
        
        This method provides two processing modes:
        1. Standard mode (use_batch_api=False): Uses parallel invoke for immediate results
        2. Batch API mode (use_batch_api=True): Uses OpenAI's Batch API for 50% cost savings
        
        Args:
            inputs: List of inputs to process in batch.
            config: Configuration for the batch processing.
            return_exceptions: Whether to return exceptions instead of raising them.
            use_batch_api: If True, use OpenAI's Batch API for cost savings with polling.
                          If False (default), use standard parallel processing for immediate results.
            **kwargs: Additional parameters to pass to the underlying model.
            
        Returns:
            List of BaseMessage objects corresponding to the inputs.
            
        Raises:
            BatchError: If batch processing fails (when use_batch_api=True).
            
        Note:
            **Cost vs Latency Tradeoff:**
            - use_batch_api=False: Immediate results, standard API pricing
            - use_batch_api=True: 50% cost savings, asynchronous processing with polling
            
        Example:
            .. code-block:: python
            
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage
                
                llm = ChatOpenAI()
                inputs = [
                    [HumanMessage(content="What is 2+2?")],
                    [HumanMessage(content="What is the capital of France?")],
                ]
                
                # Standard processing (immediate results)
                results = llm.batch(inputs)
                
                # Batch API processing (50% cost savings, polling required)
                results = llm.batch(inputs, use_batch_api=True)
        """
        if use_batch_api:
            # Convert inputs to messages_list format expected by batch_create
            messages_list = []
            for input_item in inputs:
                if isinstance(input_item, list):
                    # Already a list of messages
                    messages_list.append(input_item)
                else:
                    # Convert single input to list of messages
                    messages = self._convert_input_to_messages(input_item)
                    messages_list.append(messages)
            
            # Create batch job and poll for results
            batch_id = self.batch_create(messages_list, **kwargs)
            chat_results = self.batch_retrieve(batch_id)
            
            # Convert ChatResult objects to BaseMessage objects
            return [result.generations[0].message for result in chat_results]
        else:
            # Use the parent class's standard batch implementation
            return super().batch(
                inputs=inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )

    def _convert_input_to_messages(self, input_item: LanguageModelInput) -> List[BaseMessage]:
        """Convert various input formats to a list of BaseMessage objects."""
        if isinstance(input_item, list):
            # Already a list of messages
            return input_item
        elif isinstance(input_item, BaseMessage):
            # Single message
            return [input_item]
        elif isinstance(input_item, str):
            # String input - convert to HumanMessage
            from langchain_core.messages import HumanMessage
            return [HumanMessage(content=input_item)]
        elif hasattr(input_item, 'to_messages'):
            # PromptValue or similar
            return input_item.to_messages()
        else:
            # Try to convert to string and then to HumanMessage
            from langchain_core.messages import HumanMessage
            return [HumanMessage(content=str(input_item))]

{indent}'''
    
    # Insert the method
    new_content = content[:insert_pos] + method + content[insert_pos:]
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print('Successfully added batch() method override to BaseChatOpenAI class')
    return True

if __name__ == "__main__":
    add_batch_override()
