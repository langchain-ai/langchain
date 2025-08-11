#!/usr/bin/env python3
"""Script to fix trailing whitespace in batch.py."""

with open('langchain_openai/chat_models/batch.py', 'r') as f:
    content = f.read()

# Remove all trailing whitespace and ensure single newline at end
content = content.rstrip() + '\n'

with open('langchain_openai/chat_models/batch.py', 'w') as f:
    f.write(content)

print("Fixed trailing whitespace in batch.py")
