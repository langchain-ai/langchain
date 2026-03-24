"""File tools — write markdown output to local directories.

Content is always written to local files for human review before publishing.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Output root is relative to the repo root
OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "output"


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug.

    Args:
        text: The text to slugify.

    Returns:
        A lowercase, hyphen-separated slug string.
    """
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def write_brief(keyword: str, content: str) -> str:
    """Write a content brief to the briefs output directory.

    Args:
        keyword: The target keyword (used for filename).
        content: The markdown content to write.

    Returns:
        The absolute file path of the written brief.
    """
    out_dir = OUTPUT_ROOT / "briefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(keyword)}.md"
    filepath = out_dir / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info("Brief written to %s", filepath)
    return str(filepath)


def write_draft(keyword: str, content: str) -> str:
    """Write a content draft to the drafts output directory.

    Args:
        keyword: The target keyword (used for filename).
        content: The markdown content to write.

    Returns:
        The absolute file path of the written draft.
    """
    out_dir = OUTPUT_ROOT / "drafts"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(keyword)}.md"
    filepath = out_dir / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info("Draft written to %s", filepath)
    return str(filepath)


def write_report(name: str, content: str) -> str:
    """Write a report to the reports output directory.

    Args:
        name: Report name (used for filename, e.g. ``seo-report-2026-03-24``).
        content: The markdown content to write.

    Returns:
        The absolute file path of the written report.
    """
    out_dir = OUTPUT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(name)}.md"
    filepath = out_dir / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info("Report written to %s", filepath)
    return str(filepath)


def read_brief(keyword: str) -> str | None:
    """Read a content brief from the briefs output directory.

    Args:
        keyword: The keyword slug to look up.

    Returns:
        The markdown content, or None if the file does not exist.
    """
    filepath = OUTPUT_ROOT / "briefs" / f"{_slugify(keyword)}.md"
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return None


def list_output_files(subdir: str = "briefs") -> list[str]:
    """List all files in an output subdirectory.

    Args:
        subdir: The subdirectory name (``briefs``, ``drafts``, or ``reports``).

    Returns:
        List of filenames.
    """
    out_dir = OUTPUT_ROOT / subdir
    if not out_dir.exists():
        return []
    return sorted(f.name for f in out_dir.iterdir() if f.is_file())
