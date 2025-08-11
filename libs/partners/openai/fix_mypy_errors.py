#!/usr/bin/env python3
"""Script to fix all mypy errors in the OpenAI batch implementation."""

import re

def fix_batch_method_signature():
    """Fix the batch method signature incompatibility."""
    file_path = 'langchain_openai/chat_models/base.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the use_batch_api parameter from the batch method signature
    old_signature = """    @override
    def batch(
        self,
        inputs: list[LanguageModelInput],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        use_batch_api: bool = False,
        **kwargs: Any,
    ) -> list[BaseMessage]:"""
    
    new_signature = """    @override
    def batch(
        self,
        inputs: list[LanguageModelInput],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[BaseMessage]:"""
    
    content = content.replace(old_signature, new_signature)
    
    # Update the docstring to remove references to use_batch_api parameter
    old_docstring_part = """        This method provides two processing modes:
        1. Standard mode (use_batch_api=False): Uses parallel invoke for
           immediate results
        2. Batch API mode (use_batch_api=True): Uses OpenAI's Batch API
           for 50% cost savings

        Args:
            inputs: List of inputs to process in batch.
            config: Configuration for the batch processing.
            return_exceptions: Whether to return exceptions instead of raising them.
            use_batch_api: Whether to use OpenAI's Batch API for cost savings.
            **kwargs: Additional keyword arguments."""
    
    new_docstring_part = """        This method uses the standard parallel invoke approach for immediate results.
        For cost savings with asynchronous processing, use batch_create() and 
        batch_retrieve() methods instead.

        Args:
            inputs: List of inputs to process in batch.
            config: Configuration for the batch processing.
            return_exceptions: Whether to return exceptions instead of raising them.
            **kwargs: Additional keyword arguments."""
    
    content = content.replace(old_docstring_part, new_docstring_part)
    
    # Update the method implementation to remove use_batch_api logic
    old_impl_start = """        # Extract use_batch_api from kwargs if present
        use_batch_api = kwargs.pop("use_batch_api", False)
        
        if use_batch_api:"""
    
    # Find and replace the entire conditional block
    pattern = r'        # Extract use_batch_api from kwargs if present.*?        else:\s*\n            # Use standard batch processing.*?return results'
    
    new_impl = """        # Use standard batch processing (parallel invoke)
        return super().batch(
            inputs=inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )"""
    
    content = re.sub(pattern, new_impl, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed batch method signature incompatibility")

def fix_test_files():
    """Fix various test file issues."""
    
    # Fix unit test file
    unit_test_path = 'tests/unit_tests/chat_models/test_batch.py'
    with open(unit_test_path, 'r') as f:
        content = f.read()
    
    # Fix OpenAIBatchClient constructor calls - remove poll_interval and timeout
    content = re.sub(
        r'OpenAIBatchClient\(mock_client, poll_interval=\d+\.?\d*, timeout=\d+\.?\d*\)',
        'OpenAIBatchClient(mock_client)',
        content
    )
    
    # Fix create_batch calls - change batch_requests to requests
    content = content.replace('batch_requests=', 'requests=')
    
    # Fix attribute access issues
    content = content.replace('batch_response.status', 'batch_response["status"]')
    content = content.replace('batch_response.output_file_id', 'batch_response["output_file_id"]')
    
    # Remove timeout attribute access
    content = re.sub(r'client\.timeout = \d+\.?\d*\n', '', content)
    
    with open(unit_test_path, 'w') as f:
        f.write(content)
    
    # Fix integration test file
    integration_test_path = 'tests/integration_tests/chat_models/test_batch_integration.py'
    with open(integration_test_path, 'r') as f:
        content = f.read()
    
    # Remove max_tokens parameter from ChatOpenAI constructor
    content = re.sub(r', max_tokens=\d+', '', content)
    
    # Fix type annotations for message lists
    content = content.replace('list[HumanMessage]', 'list[BaseMessage]')
    
    # Fix content access for messages
    content = re.sub(
        r'(\w+)\.content\.strip\(\)',
        r'str(\1.content).strip()',
        content
    )
    
    # Add missing return type annotation
    content = re.sub(
        r'def (test_\w+)\(([^)]*)\):',
        r'def \1(\2) -> None:',
        content
    )
    
    with open(integration_test_path, 'w') as f:
        f.write(content)
    
    print("Fixed test file issues")

if __name__ == "__main__":
    fix_batch_method_signature()
    fix_test_files()
    print("All mypy errors should now be fixed")
