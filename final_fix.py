#!/usr/bin/env python3
"""Final comprehensive script to fix all remaining linting issues."""

import re

def fix_base_py_final():
    """Fix all remaining issues in base.py"""
    file_path = '/home/daytona/langchain/libs/partners/openai/langchain_openai/chat_models/base.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add missing imports - find a good place to add them
    # Look for existing langchain_core imports and add after them
    if 'from langchain_core.runnables import RunnablePassthrough' in content:
        content = content.replace(
            'from langchain_core.runnables import RunnablePassthrough',
            'from langchain_core.runnables import RunnablePassthrough\nfrom langchain_core.runnables.config import RunnableConfig\nfrom typing_extensions import override'
        )
    
    # Fix long lines in docstrings
    content = content.replace(
        '         2. Batch API mode (use_batch_api=True): Uses OpenAI\'s Batch API for 50% cost savings',
        '         2. Batch API mode (use_batch_api=True): Uses OpenAI\'s Batch API for\n            50% cost savings'
    )
    
    content = content.replace(
        '            use_batch_api: If True, use OpenAI\'s Batch API for cost savings with polling.',
        '            use_batch_api: If True, use OpenAI\'s Batch API for cost savings\n                          with polling.'
    )
    
    content = content.replace(
        '                          If False (default), use standard parallel processing for immediate results.',
        '                          If False (default), use standard parallel processing\n                          for immediate results.'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed base.py final issues")

def fix_batch_py_final():
    """Fix all remaining issues in batch.py"""
    file_path = '/home/daytona/langchain/libs/partners/openai/langchain_openai/chat_models/batch.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix remaining List references
    content = re.sub(r'list\[List\[BaseMessage\]\]', 'list[list[BaseMessage]]', content)
    
    # Fix unused variable
    content = content.replace(
        'batch = self.client.batches.cancel(batch_id)',
        '_ = self.client.batches.cancel(batch_id)'
    )
    
    # Fix long lines
    content = content.replace(
        '    High-level processor for managing OpenAI Batch API lifecycle with LangChain integration.',
        '    High-level processor for managing OpenAI Batch API lifecycle with\n    LangChain integration.'
    )
    
    content = content.replace(
        '                    f"Batch {batch_id} failed with status {batch_info[\'status\']}"',
        '                    f"Batch {batch_id} failed with status "\n                    f"{batch_info[\'status\']}"'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed batch.py final issues")

def fix_unit_tests_final():
    """Fix remaining unit test issues"""
    file_path = '/home/daytona/langchain/libs/partners/openai/tests/unit_tests/chat_models/test_batch.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the specific problematic test and fix it properly
    # Look for the pattern where results is undefined
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for the problematic pattern
        if '_ = self.llm.batch(inputs, use_batch_api=True)' in line:
            # Check if the next few lines reference 'results'
            next_lines = lines[i+1:i+5] if i+5 < len(lines) else lines[i+1:]
            if any('assert len(results)' in next_line or 'for i, result in enumerate(results)' in next_line for next_line in next_lines):
                # Replace the assignment to actually capture results
                fixed_lines.append(line.replace('_ = self.llm.batch(inputs, use_batch_api=True)', 'results = self.llm.batch(inputs, use_batch_api=True)'))
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed unit tests final issues")

if __name__ == "__main__":
    print("Running final comprehensive fixes...")
    fix_base_py_final()
    fix_batch_py_final()
    fix_unit_tests_final()
    print("All final fixes completed!")
