"""Utility validators for inputs."""
from __future__ import annotations

from urllib.parse import urlparse

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"}
INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com"}


def guess_platform(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        host = host.lower()
        if any(h in host for h in YOUTUBE_HOSTS):
            return "youtube"
        if any(h in host for h in INSTAGRAM_HOSTS):
            return "instagram"
    except Exception:
        return None
    return None
