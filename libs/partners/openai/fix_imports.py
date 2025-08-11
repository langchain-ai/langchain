#!/usr/bin/env python3
"""Fix remaining import and formatting issues."""


def fix_base_py():
    """Fix base.py imports and line length issues."""
    with open("langchain_openai/chat_models/base.py") as f:
        content = f.read()

    # Add missing imports
    content = content.replace(
        "from typing_extensions import Self",
        "from typing_extensions import Self, override",
    )

    content = content.replace(
        "from langchain_core.runnables.config import run_in_executor",
        "from langchain_core.runnables.config import RunnableConfig, run_in_executor",
    )

    # Fix long lines in docstrings
    content = content.replace(
        "         2. Batch API mode (use_batch_api=True): Uses OpenAI's Batch API for 50% cost savings",
        "         2. Batch API mode (use_batch_api=True): Uses OpenAI's Batch API for\n            50% cost savings",
    )

    with open("langchain_openai/chat_models/base.py", "w") as f:
        f.write(content)
    print("Fixed base.py")


def fix_batch_py():
    """Fix batch.py line length issues."""
    with open("langchain_openai/chat_models/batch.py") as f:
        content = f.read()

    # Fix long lines
    content = content.replace(
        "    50% cost savings compared to the standard API in exchange for asynchronous processing.",
        "    50% cost savings compared to the standard API in exchange for\n    asynchronous processing.",
    )

    content = content.replace(
        '                    f"Batch {batch_id} timed out after {timeout} seconds. Current status: {status}",',
        '                    f"Batch {batch_id} timed out after {timeout} seconds. "\n                    f"Current status: {status}",',
    )

    content = content.replace(
        "                     f\"Batch {batch_id} is not completed. Current status: {batch_info['status']}\",",
        '                     f"Batch {batch_id} is not completed. "\n                     f"Current status: {batch_info[\'status\']}",',
    )

    with open("langchain_openai/chat_models/batch.py", "w") as f:
        f.write(content)
    print("Fixed batch.py")


if __name__ == "__main__":
    fix_base_py()
    fix_batch_py()
    print("All fixes applied!")
