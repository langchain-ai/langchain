#!/usr/bin/env python3
"""Script to check for reStructuredText (RST) syntax in Python files.

This script checks for RST-specific syntax patterns that should not be used
in docstrings. The project uses Markdown/MkDocs Material syntax instead.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# RST patterns to detect (pattern, description)
RST_PATTERNS: List[Tuple[str, str]] = [
    (r"\.\.\s+code-block::", "RST code-block directive (use ```language instead)"),
    (r":func:", "RST :func: role (use regular Markdown links)"),
    (r":class:", "RST :class: role (use regular Markdown links)"),
    (r":meth:", "RST :meth: role (use regular Markdown links)"),
    (
        r"\.\.\s+deprecated::",
        "RST deprecated directive (use !!! deprecated admonition)",
    ),
    (r"\.\.\s+dropdown::", "RST dropdown directive (use ??? admonitions)"),
    (r"\.\.\s+versionadded::", "RST versionadded directive"),
    (
        r"\.\.\s+versionchanged::",
        "RST versionchanged directive",
    ),
    (r"\.\.\s+warning::", "RST warning directive (use !!! warning admonition)"),
    (r"\.\.\s+important::", "RST important directive (use !!! important admonition)"),
    (r"\.\.\s+note::", "RST note directive (use !!! note admonition)"),
    (r":private:", "RST :private: role (prefix name with underscore instead)"),
    (r"`[^`]+<[^>]+>`__", "RST anonymous hyperlink (use [text](link) instead)"),
    (r"`[^`]+<[^>]+>`_", "RST named hyperlink (use [text](link) instead)"),
]


def check_file_for_rst(file_path: Path) -> List[Tuple[int, str, str]]:
    """Check a single file for RST syntax patterns.

    Args:
        file_path: Path to the file to check.

    Returns:
        List of tuples containing (line_number, matched_text, description).
    """
    violations: List[Tuple[int, str, str]] = []

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, start=1):
            for pattern, description in RST_PATTERNS:
                matches = re.finditer(pattern, line)
                for match in matches:
                    violations.append((line_num, match.group(0), description))

    except (UnicodeDecodeError, PermissionError):
        # Skip files that can't be read as text
        pass

    return violations


def main() -> int:
    """Main entry point for the script.

    Returns:
        Exit code: 0 if no violations found, 1 if violations detected.
    """
    # Get files from command line arguments
    files = sys.argv[1:]

    if not files:
        print("No files provided to check.", file=sys.stderr)
        return 0

    has_violations = False

    for file_str in files:
        file_path = Path(file_str)

        if file_path.suffix != ".py":
            # Only check Python files
            continue

        if not file_path.exists():
            continue

        violations = check_file_for_rst(file_path)

        if violations:
            has_violations = True
            print(f"\nRST syntax detected in: {file_path}")
            for line_num, matched_text, description in violations:
                print(f"  Line {line_num}: {matched_text}")
                print(f"    → {description}")

    if has_violations:
        print(
            "\nRST syntax is not allowed. Please use Markdown/MkDocs Material syntax instead."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
