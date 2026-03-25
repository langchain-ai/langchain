"""Utility functions for the Plasmate LangChain integration."""

import asyncio
import json
import subprocess
from typing import Any, Optional

from langchain_core._security._ssrf_protection import validate_safe_url


def _get_plasmate_bin() -> str:
    """Find the plasmate binary."""
    import shutil

    path = shutil.which("plasmate")
    if path:
        return path
    raise FileNotFoundError(
        "plasmate binary not found. Install with:\n"
        "  cargo install plasmate\n"
        "  # or: curl -fsSL https://plasmate.app/install.sh | sh"
    )


def fetch_som(url: str, plasmate_bin: Optional[str] = None) -> dict[str, Any]:
    """Fetch a URL and return parsed SOM output.

    Args:
        url: The URL to fetch.
        plasmate_bin: Path to plasmate binary. Auto-detected if not provided.

    Returns:
        Parsed SOM dictionary.
    """
    validated_url = validate_safe_url(url, allow_private=False, allow_http=True)
    bin_path = plasmate_bin or _get_plasmate_bin()
    result = subprocess.run(
        [bin_path, "fetch", validated_url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"plasmate fetch failed for {validated_url}: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


async def fetch_som_async(
    url: str, plasmate_bin: Optional[str] = None
) -> dict[str, Any]:
    """Async version of fetch_som."""
    validated_url = validate_safe_url(url, allow_private=False, allow_http=True)
    bin_path = plasmate_bin or _get_plasmate_bin()
    proc = await asyncio.create_subprocess_exec(
        bin_path,
        "fetch",
        validated_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(
            f"plasmate fetch failed for {validated_url}: {stderr.decode().strip()}"
        )
    return json.loads(stdout.decode())


def som_to_text(som_data: dict[str, Any]) -> str:
    """Convert SOM data to a readable text representation for LLM consumption."""
    from som_parser import parse_som, to_markdown

    som = parse_som(som_data)
    return to_markdown(som)


def som_to_context(som_data: dict[str, Any]) -> str:
    """Convert SOM data to a structured context string with metadata."""
    from som_parser import (
        parse_som,
        get_interactive_elements,
        get_links,
        get_text,
    )

    som = parse_som(som_data)
    lines = [
        f"# {som.title}",
        f"URL: {som.url}",
        "",
    ]

    interactive = get_interactive_elements(som)
    if interactive:
        lines.append(f"## Interactive Elements ({len(interactive)})")
        for el in interactive[:30]:
            actions = ", ".join(a.value for a in (el.actions or []))
            label = el.text or el.label or ""
            if el.attrs and el.attrs.placeholder:
                label = label or el.attrs.placeholder
            lines.append(f"  [{el.id}] {el.role.value}: {label} ({actions})")
        lines.append("")

    text = get_text(som)
    if text:
        lines.append("## Content")
        lines.append(text[:3000])
        if len(text) > 3000:
            lines.append(f"... ({len(text) - 3000} more characters)")

    meta = som.meta
    ratio = meta.html_bytes / max(meta.som_bytes, 1)
    lines.append("")
    lines.append(
        f"[Compression: {ratio:.1f}x, "
        f"{meta.element_count} elements, "
        f"{meta.interactive_count} interactive]"
    )

    return "\n".join(lines)
