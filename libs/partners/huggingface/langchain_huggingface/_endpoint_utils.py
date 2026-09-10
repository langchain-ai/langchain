"""Helpers shared by the Hugging Face inference endpoint integrations."""

from __future__ import annotations

from urllib.parse import urlparse


def _is_huggingface_hosted_url(url: str | None) -> bool:
    """True if url is HF-hosted (huggingface.co or hf.space)."""
    if not url:
        return False
    hostname = (urlparse(url).hostname or "").lower()
    return (
        hostname == "huggingface.co"
        or hostname == "hf.space"
        or hostname.endswith((".huggingface.co", ".hf.space"))
    )
