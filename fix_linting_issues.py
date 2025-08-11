#!/usr/bin/env python3
"""Script to fix linting issues in the OpenAI batch API implementation."""

import re

def fix_base_py_type_annotations():
    """Fix type annotations to use modern syntax"""
    file_path = '/home/daytona/langchain/libs/partners/openai/langchain_openai/chat_models/base.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove Dict and List from imports since we'll use lowercase versions
    content = re.sub(r'(\s+)Dict,\n', '', content)
    content = re.sub(r'(\s+)List,\n', '', content)
    
    # Replace type annotations
    content = re.sub(r'List\[List\[BaseMessage\]\]', 'list[list[BaseMessage]]', content)
    content = re.sub(r'Dict\[str, str\]', 'dict[str, str]', content)
    content = re.sub(r'List\[ChatResult\]', 'list[ChatResult]', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed type annotations in base.py")

def fix_integration_test_issues():
    """Fix all issues in integration tests"""
    file_path = '/home/daytona/langchain/libs/partners/openai/tests/integration_tests/chat_models/test_batch_integration.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix long lines by breaking them
    content = content.replace(
        'content="What is the capital of France? Answer with just the city name."',
        'content="What is the capital of France? Answer with just the city name."'
    )
    
    content = content.replace(
        'content="What is the smallest planet? Answer with just the planet name."',
        'content="What is the smallest planet? Answer with just the planet name."'
    )
    
    # Fix unused variables by using underscore
    content = re.sub(r'sequential_time = time\.time\(\) - start_sequential', 
                     '_ = time.time() - start_sequential', content)
    content = re.sub(r'batch_time = time\.time\(\) - start_batch', 
                     '_ = time.time() - start_batch', content)
    
    # Fix long comment line
    content = content.replace(
        '        # Log timing comparison        # Note: Batch API will typically be slower for small batches due to polling,',
        '        # Note: Batch API will typically be slower for small batches due to polling,'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed integration test issues")

def fix_unit_test_issues():
    """Fix all issues in unit tests"""
    file_path = '/home/daytona/langchain/libs/partners/openai/tests/unit_tests/chat_models/test_batch.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the test that has undefined results variable
    # Find the test method and fix it properly
    pattern = r'(\s+)_ = self\.llm\.batch\(inputs, use_batch_api=True\)\s*\n\s*# Verify conversion happened\s*\n\s*assert len\(results\) == num_requests'
    replacement = r'\1results = self.llm.batch(inputs, use_batch_api=True)\n\1# Verify conversion happened\n\1assert len(results) == num_requests'
    content = re.sub(pattern, replacement, content)
    
    # Fix other undefined results references
    content = re.sub(r'(\s+)_ = self\.llm\.batch\(inputs, use_batch_api=True\)\s*\n(\s+)assert len\(results\) == 2',
                     r'\1results = self.llm.batch(inputs, use_batch_api=True)\n\2assert len(results) == 2', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed unit test issues")

if __name__ == "__main__":
    print("Fixing linting issues...")
    fix_base_py_type_annotations()
    fix_integration_test_issues()
    fix_unit_test_issues()
    print("All linting issues fixed!")

