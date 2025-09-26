#!/usr/bin/env python3
"""
Convert markdown elements to Sphinx directives.

This script processes files and converts:

1. Markdown-style code blocks like:
    ```python
    code here
    ```
   To Sphinx code-block directives like:
    .. code-block:: python

        code here

2. Markdown-style admonitions like:
    !!! note
   To Sphinx directives like:
    .. deprecated:: 0.2.0
    .. warning:: Weird behavior
    .. note::

Usage:
    python convert_markdown_codeblocks.py [file_or_directory]
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple


def convert_markdown_admonitions(content: str) -> str:
    """
    Convert markdown-style admonitions to Sphinx directives.

    Args:
        content: File content to process

    Returns:
        Content with markdown admonitions converted to Sphinx directives
    """
    if not content or "!!!" not in content:
        return content

    lines = content.split("\n")
    result_lines = []

    for line in lines:
        # Match patterns like "!!! deprecated 0.2.0", "!!! warning", etc.
        match = re.match(r"^(\s*)!!!\s+(\w+)(?:\s+(.+))?\s*$", line)
        if match:
            leading_indent = match.group(1)
            directive_type = match.group(2)
            directive_arg = match.group(3) or ""

            # Convert to Sphinx directive format
            if directive_arg:
                result_lines.append(
                    f"{leading_indent}.. {directive_type}:: {directive_arg}"
                )
            else:
                result_lines.append(f"{leading_indent}.. {directive_type}::")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def convert_markdown_codeblock(content: str) -> str:
    """
    Convert markdown code blocks to Sphinx code-block directives.

    Args:
        content: File content to process

    Returns:
        Content with markdown code blocks converted to Sphinx directives
    """
    lines = content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line starts a code block
        match = re.match(r"^(\s*)```(\w*)\s*$", line)
        if match:
            leading_indent = match.group(1)
            language = match.group(2) or ""

            # Find the closing ``` with same indentation
            closing_idx = None
            for j in range(i + 1, len(lines)):
                if re.match(rf"^{re.escape(leading_indent)}```\s*$", lines[j]):
                    closing_idx = j
                    break

            if closing_idx is not None:
                # Extract code content
                code_lines = lines[i + 1 : closing_idx]

                # Build Sphinx code-block directive
                if language:
                    result_lines.append(f"{leading_indent}.. code-block:: {language}")
                else:
                    result_lines.append(f"{leading_indent}.. code-block::")

                result_lines.append("")  # Empty line after directive

                # Add indented code content
                # For Sphinx, code should be indented 4 spaces from the directive
                # First, normalize the code by removing common leading whitespace
                if code_lines:
                    # Find the minimum indentation of non-empty lines
                    non_empty_lines = [line for line in code_lines if line.strip()]
                    if non_empty_lines:
                        min_indent = min(
                            len(line) - len(line.lstrip()) for line in non_empty_lines
                        )
                        # Remove the common leading whitespace
                        normalized_lines = []
                        for line in code_lines:
                            if line.strip():
                                # Remove common indentation but preserve relative indentation
                                normalized_lines.append(
                                    line[min_indent:]
                                    if len(line) > min_indent
                                    else line.lstrip()
                                )
                            else:
                                normalized_lines.append("")
                    else:
                        normalized_lines = code_lines
                else:
                    normalized_lines = []

                # Add the normalized code with Sphinx indentation (4 spaces from directive)
                for code_line in normalized_lines:
                    if code_line.strip():
                        result_lines.append(f"{leading_indent}    {code_line}")
                    else:
                        result_lines.append(leading_indent.rstrip())

                # Skip to after the closing ```
                i = closing_idx + 1
            else:
                # No matching closing ```, treat as regular line
                result_lines.append(line)
                i += 1
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def should_process_file(file_path: Path) -> bool:
    """
    Determine if a file should be processed based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        True if the file should be processed
    """
    # Process Python files, Markdown files, MDX files, and RST files
    extensions = {".py", ".md", ".mdx", ".rst", ".ipynb"}
    return file_path.suffix.lower() in extensions


def has_markdown_elements(content: str) -> bool:
    """
    Check if content contains markdown code blocks or admonitions.

    Args:
        content: File content to check

    Returns:
        True if content contains markdown elements that need conversion
    """
    # Check for markdown code blocks or admonitions
    return "```" in content or "!!!" in content


def process_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Process a single file to convert markdown code blocks.

    Args:
        file_path: Path to the file to process
        dry_run: If True, don't write changes, just report what would be done

    Returns:
        Tuple of (success, message)
    """
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Check if file has markdown elements
        if not has_markdown_elements(original_content):
            return True, f"No markdown elements found in {file_path}"

        # Convert the content - admonitions first, then code blocks
        converted_content = convert_markdown_admonitions(original_content)
        converted_content = convert_markdown_codeblock(converted_content)

        if converted_content == original_content:
            return True, f"No changes needed in {file_path}"

        if dry_run:
            return True, f"Would convert markdown elements in {file_path}"

        # Write the converted content back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(converted_content)

        return True, f"Converted markdown elements in {file_path}"

    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"


def process_directory(directory: Path, dry_run: bool = False) -> List[str]:
    """
    Process all files in a directory recursively.

    Args:
        directory: Directory to process
        dry_run: If True, don't write changes, just report what would be done

    Returns:
        List of processing messages
    """
    messages = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common build/cache directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d not in {"__pycache__", "node_modules", "_build"}
        ]

        for file_name in files:
            file_path = Path(root) / file_name

            if should_process_file(file_path):
                success, message = process_file(file_path, dry_run)
                messages.append(message)
                if not success:
                    print(f"ERROR: {message}")
                else:
                    print(message)

    return messages


def main():
    """Main function to run the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert markdown elements (code blocks and admonitions) to Sphinx directives"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to process (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )

    args = parser.parse_args()

    target_path = Path(args.path)

    if not target_path.exists():
        print(f"Error: Path {target_path} does not exist")
        return 1

    print(f"Processing: {target_path}")
    if args.dry_run:
        print("DRY RUN - No files will be modified")
    print()

    if target_path.is_file():
        if should_process_file(target_path):
            success, message = process_file(target_path, args.dry_run)
            print(message)
            return 0 if success else 1
        else:
            print(f"File {target_path} is not a supported file type")
            return 1
    else:
        messages = process_directory(target_path, args.dry_run)

        # Summary
        total_files = len([msg for msg in messages if "Error processing" not in msg])
        converted_files = len(
            [
                msg
                for msg in messages
                if "Converted markdown elements" in msg or "Would convert" in msg
            ]
        )
        errors = len([msg for msg in messages if "Error processing" in msg])

        print(f"\nSummary:")
        print(f"  Total files processed: {total_files}")
        print(f"  Files with conversions: {converted_files}")
        if errors > 0:
            print(f"  Errors: {errors}")

        return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
