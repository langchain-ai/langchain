#!/usr/bin/env python3
"""Script to add missing return type annotations to test functions."""

import re

def fix_test_file(file_path):
    """Fix missing return type annotations in a test file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match test functions without return type annotations
    pattern = r'(def test_[^(]*\([^)]*\)):'
    
    def replace_func(match):
        func_def = match.group(1)
        return f"{func_def} -> None:"
    
    # Replace all test function definitions
    new_content = re.sub(pattern, replace_func, content)
    
    # Also fix other common test function patterns
    patterns = [
        (r'(def mock_[^(]*\([^)]*\)):', r'\1 -> None:'),
        (r'(def setup_[^(]*\([^)]*\)):', r'\1 -> None:'),
        (r'(def teardown_[^(]*\([^)]*\)):', r'\1 -> None:'),
    ]
    
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, new_content)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed return type annotations in {file_path}")

if __name__ == "__main__":
    # Fix both test files
    fix_test_file('tests/unit_tests/chat_models/test_batch.py')
    fix_test_file('tests/integration_tests/chat_models/test_batch_integration.py')
