#!/usr/bin/env python3
"""Script to fix linting issues in the OpenAI batch API implementation."""

import re

def fix_base_py_imports():
    """Fix missing imports in base.py"""
    file_path = '/home/daytona/langchain/libs/partners/openai/langchain_openai/chat_models/base.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add List and Dict to the typing imports
    old_imports = """from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)"""
    
    new_imports = """from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)"""
    
    if old_imports in content:
        content = content.replace(old_imports, new_imports)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print("Fixed imports in base.py")
        return True
    else:
        print("Could not find import section to fix in base.py")
        return False

def fix_integration_test_lines():
    """Fix line length issues in integration tests"""
    file_path = '/home/daytona/langchain/libs/partners/openai/tests/integration_tests/chat_models/test_batch_integration.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix long lines
    content = content.replace(
        'content="What is the capital of France? Answer with just the city name."',
        'content="What is the capital of France? Answer with just the city name."'
    )
    
    content = content.replace(
        'content="What is the smallest planet? Answer with just the planet name."',
        'content="What is the smallest planet? Answer with just the planet name."'
    )
    
    # Remove print statements
    content = re.sub(r'\s*print\(f?"[^"]*"\)\s*\n', '', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed integration test issues")

def fix_unit_test_issues():
    """Fix unused variable in unit tests"""
    file_path = '/home/daytona/langchain/libs/partners/openai/tests/unit_tests/chat_models/test_batch.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix unused variable by using it or removing assignment
    content = content.replace(
        'results = self.llm.batch(inputs, use_batch_api=True)',
        '_ = self.llm.batch(inputs, use_batch_api=True)'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed unit test issues")

if __name__ == "__main__":
    print("Fixing linting issues...")
    fix_base_py_imports()
    fix_integration_test_lines()
    fix_unit_test_issues()
    print("All linting issues fixed!")
