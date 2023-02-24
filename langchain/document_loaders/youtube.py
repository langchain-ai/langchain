"""Loader that loads YouTube transcript."""
from __future__ import annotations

from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class YoutubeLoader(BaseLoader):
    """Loader that loads Youtube transcripts."""

    def __init__(
        self, video_id: str, add_video_info: bool = False, language: str = "en"
    ):
        """Initialize with YouTube video ID."""
        self.video_id = video_id
        self.add_video_info = add_video_info
        self.language = language

    @classmethod
    def from_youtube_url(cls, youtube_url: str, **kwargs: Any) -> YoutubeLoader:
        """Parse out video id from YouTube url."""
        video_id = youtube_url.split("youtube.com/watch?v=")[-1]
        return cls(video_id, **kwargs)

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "Could not import youtube_transcript_api python package. "
                "Please it install it with `pip install youtube-transcript-api`."
            )

        metadata = {"source": self.video_id}

        if self.add_video_info:
            # Get more video meta info
            # Such as title, description, thumbnail url, publish_date
            video_info = self._get_video_info()
            metadata.update(video_info)

        transcript_pieces = YouTubeTranscriptApi.get_transcript(
            self.video_id, languages=(self.language,)
        )
        transcript = " ".join([t["text"].strip(" ") for t in transcript_pieces])

        return [Document(page_content=transcript, metadata=metadata)]

    def _get_video_info(self) -> dict:
        """Get important video information.

        Components are:
            - title
            - description
            - thumbnail url,
            - publish_date
            - channel_author
            - and more.
        """
        try:
            from pytube import YouTube

        except ImportError:
            raise ImportError(
                "Could not import pytube python package. "
                "Please it install it with `pip install pytube`."
            )
        yt = YouTube(f"https://www.youtube.com/watch?v={self.video_id}")
        video_info = {
            "title": yt.title,
            "description": yt.description,
            "view_count": yt.views,
            "thumbnail_url": yt.thumbnail_url,
            "publish_date": yt.publish_date,
            "length": yt.length,
            "author": yt.author,
        }
        return video_info
