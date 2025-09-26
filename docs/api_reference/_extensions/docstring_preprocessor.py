"""Sphinx extension to preprocess docstrings before napoleon conversion.

This extension converts markdown code blocks to Sphinx code-block directives
in docstrings before the napoleon extension processes them.
"""

import re
from typing import Any, Dict, List, Optional

from sphinx.application import Sphinx
from sphinx.ext.autodoc import Documenter


def convert_markdown_admonitions_in_docstring(docstring: str) -> str:
    """
    Convert markdown-style admonitions to Sphinx directives.

    Args:
        docstring: The docstring content to process

    Returns:
        Docstring with markdown admonitions converted to Sphinx directives
    """
    if not docstring or "!!!" not in docstring:
        return docstring

    lines = docstring.split("\n")
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


def convert_markdown_codeblocks_in_docstring(docstring: str) -> str:
    """
    Convert markdown code blocks to Sphinx code-block directives in a docstring.

    Args:
        docstring: The docstring content to process

    Returns:
        Docstring with markdown code blocks converted to Sphinx directives
    """
    if not docstring or "```" not in docstring:
        return docstring

    lines = docstring.split("\n")
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
                # Normalize code by removing common leading whitespace
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


def process_docstring(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Dict[str, Any],
    lines: List[str],
) -> None:
    """
    Process docstring to convert markdown elements to Sphinx directives.

    This function is called by Sphinx's autodoc-process-docstring event
    before napoleon processes the docstring.
    """
    if not lines:
        return

    # Join lines to get the full docstring
    docstring = "\n".join(lines)

    # Convert markdown admonitions first
    converted = convert_markdown_admonitions_in_docstring(docstring)

    # Convert markdown code blocks
    converted = convert_markdown_codeblocks_in_docstring(converted)

    # Only update if there were changes
    if converted != docstring:
        # Clear the original lines and replace with converted content
        lines.clear()
        lines.extend(converted.split("\n"))


def setup(app: Sphinx) -> Dict[str, Any]:
    """Setup the Sphinx extension."""
    # Connect our docstring processor to run before napoleon
    # Priority 500 ensures it runs before napoleon (which has default priority)
    app.connect("autodoc-process-docstring", process_docstring, priority=500)

    return {
        "version": "1.0.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
