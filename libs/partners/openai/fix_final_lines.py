#!/usr/bin/env python3
"""Fix the final 2 line length violations."""

# Fix base.py line 2292
with open('langchain_openai/chat_models/base.py', 'r') as f:
    content = f.read()

# Replace the specific long line in the docstring
content = content.replace(
    "         2. Batch API mode (use_batch_api=True): Uses OpenAI's Batch API for 50% cost savings",
    "         2. Batch API mode (use_batch_api=True): Uses OpenAI's Batch API\n            for 50% cost savings"
)

with open('langchain_openai/chat_models/base.py', 'w') as f:
    f.write(content)

# Fix batch.py line 212
with open('langchain_openai/chat_models/batch.py', 'r') as f:
    content = f.read()

# Replace the specific long error message
content = content.replace(
    'f"Batch {batch_id} is not completed. Current status: {batch_info[\'status\']}",',
    'f"Batch {batch_id} is not completed. "\n                     f"Current status: {batch_info[\'status\']}",',
)

with open('langchain_openai/chat_models/batch.py', 'w') as f:
    f.write(content)
