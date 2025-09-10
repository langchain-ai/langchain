#!/usr/bin/env python3
"""Script to fix type issues in test file."""

import re

# Read the test file
with open("tests/unit_tests/test_async_vectorstores.py", "r") as f:
    content = f.read()

# Replace all instances of async_client=async_client, with type ignore
content = re.sub(
    r"async_client=async_client,(?!.*# type: ignore)",
    "async_client=async_client,  # type: ignore[arg-type]",
    content
)

# Fix the all() function calls
content = re.sub(
    r"assert all\(results\[0\]\)",
    "assert all(results[0])  # type: ignore[arg-type]",
    content
)
content = re.sub(
    r"assert all\(results\[1\]\)",
    "assert all(results[1])  # type: ignore[arg-type]",
    content
)

# Write the fixed content back
with open("tests/unit_tests/test_async_vectorstores.py", "w") as f:
    f.write(content)

print("Fixed type issues in test file")
