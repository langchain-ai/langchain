"""YouTube extractor: transcript and metadata extraction utilities.

Uses youtube-transcript-api to fetch transcripts and simple HTTP calls to fetch
video metadata (title/description) via oEmbed as a lightweight approach.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import requests

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs

from content_growth_agent.tools.logger import get_logger

logger = get_logger("youtube_extractor")


def _extract_video_id(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        if parsed.hostname in ("youtu.be", "www.youtu.be"):
            return parsed.path.lstrip("/")
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]
    except Exception:
        logger.exception("Failed to parse YouTube URL")
        return None


def extract_transcript(url: str, languages: list[str] | None = None) -> str:
    """Return the transcript text for a YouTube video.

    Args:
        url: Video URL
        languages: Optional priority language list, e.g. ['en']

    Returns:
        Joined transcript text.

    Raises:
        RuntimeError if transcript cannot be retrieved.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        raise RuntimeError("Could not extract video id from URL")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        texts = [seg.get("text", "") for seg in transcript_list]
        return "\n".join(texts)
    except TranscriptsDisabled:
        logger.exception("Transcripts are disabled for video %s", video_id)
        raise RuntimeError("Transcripts are disabled for this video")
    except Exception:
        logger.exception("Failed to fetch transcript for video %s", video_id)
        raise RuntimeError("Failed to fetch transcript")


def extract_metadata(url: str) -> Dict[str, Any]:
    """Fetch basic metadata via YouTube oEmbed (title, author, thumbnail).

    Returns a dictionary with available fields.
    """
    try:
        oembed = "https://www.youtube.com/oembed"
        resp = requests.get(oembed, params={"url": url, "format": "json"}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("Failed to fetch metadata via oEmbed for %s", url)
        return {}
