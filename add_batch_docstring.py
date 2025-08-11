#!/usr/bin/env python3
"""Script to add batch API documentation to ChatOpenAI class docstring."""

import re

def add_batch_docstring():
    file_path = '/home/daytona/langchain/libs/partners/openai/langchain_openai/chat_models/base.py'
    
    # Read the base.py file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the location to insert the batch API documentation (before the closing docstring)
    pattern = r'(\s+)"""  # noqa: E501'
    match = re.search(pattern, content)
    
    if not match:
        print("Could not find insertion point")
        return False
    
    indent = match.group(1)
    insert_pos = match.start()
    
    # Define the batch API documentation to insert
    batch_docs = f'''
    .. dropdown:: Batch API for cost savings

        .. versionadded:: 0.3.7

        OpenAI's Batch API provides **50% cost savings** for non-real-time workloads by
        processing requests asynchronously. This is ideal for tasks like data processing,
        content generation, or evaluation that don't require immediate responses.

        **Cost vs Latency Tradeoff:**

        - **Standard API**: Immediate results, full pricing
        - **Batch API**: 50% cost savings, asynchronous processing (results available within 24 hours)

        **Method 1: Direct batch management**

        Use ``batch_create()`` and ``batch_retrieve()`` for full control over batch lifecycle:

        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(model="gpt-3.5-turbo")

            # Prepare multiple message sequences for batch processing
            messages_list = [
                [HumanMessage(content="Translate 'hello' to French")],
                [HumanMessage(content="Translate 'goodbye' to Spanish")],
                [HumanMessage(content="What is the capital of Italy?")],
            ]

            # Create batch job (returns immediately with batch ID)
            batch_id = llm.batch_create(
                messages_list=messages_list,
                description="Translation and geography batch",
                metadata={{"project": "multilingual_qa", "user": "analyst_1"}},
            )
            print(f"Batch created: {{batch_id}}")

            # Later, retrieve results (polls until completion)
            results = llm.batch_retrieve(
                batch_id=batch_id,
                poll_interval=60.0,  # Check every minute
                timeout=3600.0,      # 1 hour timeout
            )

            # Process results
            for i, result in enumerate(results):
                response = result.generations[0].message.content
                print(f"Response {{i+1}}: {{response}}")

        **Method 2: Enhanced batch() method**

        Use the familiar ``batch()`` method with ``use_batch_api=True`` for seamless integration:

        .. code-block:: python

            # Standard batch processing (immediate, full cost)
            inputs = [
                [HumanMessage(content="What is 2+2?")],
                [HumanMessage(content="What is 3+3?")],
            ]
            standard_results = llm.batch(inputs)  # Default: use_batch_api=False

            # Batch API processing (50% cost savings, polling)
            batch_results = llm.batch(
                inputs,
                use_batch_api=True,  # Enable cost savings
                poll_interval=30.0,  # Poll every 30 seconds
                timeout=1800.0,      # 30 minute timeout
            )

        **Batch creation with custom parameters:**

        .. code-block:: python

            # Create batch with specific model parameters
            batch_id = llm.batch_create(
                messages_list=messages_list,
                description="Creative writing batch",
                metadata={{"task_type": "content_generation"}},
                temperature=0.8,     # Higher creativity
                max_tokens=200,      # Longer responses
                top_p=0.9,          # Nucleus sampling
            )

        **Error handling and monitoring:**

        .. code-block:: python

            from langchain_openai.chat_models.batch import BatchError

            try:
                batch_id = llm.batch_create(messages_list)
                results = llm.batch_retrieve(batch_id, timeout=600.0)
            except BatchError as e:
                print(f"Batch processing failed: {{e}}")
                # Handle batch failure (retry, fallback to standard API, etc.)

        **Best practices:**

        - Use batch API for **non-urgent tasks** where 50% cost savings justify longer wait times
        - Set appropriate **timeouts** based on batch size (larger batches take longer)
        - Include **descriptive metadata** for tracking and debugging batch jobs
        - Consider **fallback strategies** for time-sensitive applications
        - Monitor batch status for **long-running jobs** to detect failures early

        **When to use Batch API:**

        ✅ **Good for:**
        - Data processing and analysis
        - Content generation at scale
        - Model evaluation and testing
        - Batch translation or summarization
        - Non-interactive applications

        ❌ **Not suitable for:**
        - Real-time chat applications
        - Interactive user interfaces
        - Time-critical decision making
        - Applications requiring immediate responses

{indent}'''
    
    # Insert the batch API documentation
    new_content = content[:insert_pos] + batch_docs + content[insert_pos:]
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print('Successfully added batch API documentation to ChatOpenAI class docstring')
    return True

if __name__ == "__main__":
    add_batch_docstring()
