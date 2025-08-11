#!/usr/bin/env python3
"""Script to fix the remaining 6 linting errors."""

def fix_base_py():
    """Fix undefined use_batch_api in base.py."""
    file_path = 'langchain_openai/chat_models/base.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and remove the entire conditional block that references use_batch_api
    lines = content.split('\n')
    new_lines = []
    skip_block = False
    
    for line in lines:
        if 'if use_batch_api:' in line:
            skip_block = True
            continue
        elif skip_block and line.strip() == '':
            continue
        elif skip_block and not line.startswith('        '):
            skip_block = False
            new_lines.append(line)
        elif not skip_block:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed undefined use_batch_api in base.py")

def fix_integration_tests():
    """Fix undefined message references in integration tests."""
    file_path = 'tests/integration_tests/chat_models/test_batch_integration.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the incorrect str(message.content) references
    content = content.replace('.str(message.content)', '.message.content')
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed undefined message references in integration tests")

if __name__ == "__main__":
    fix_base_py()
    fix_integration_tests()
    print("Fixed all remaining linting errors")
