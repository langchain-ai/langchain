"""Loader that loads YouTube transcript."""
from __future__ import annotations

from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class YoutubeLoader(BaseLoader):
    """Loader that loads Youtube transcripts."""

    def __init__(self, video_id: str):
        """Initialize with YouTube video ID."""
        self.video_id = video_id

    @classmethod
    def from_youtube_url(cls, youtube_url: str) -> YoutubeLoader:
        """Parse out video id from YouTube url."""
        video_id = youtube_url.split("youtube.com/watch?v=")[-1]
        return cls(video_id)

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ValueError(
                "Could not import youtube_transcript_api python package. "
                "Please it install it with `pip install youtube-transcript-api`."
            )
        transcript_pieces = YouTubeTranscriptApi.get_transcript(self.video_id)
        transcript = " ".join([t["text"].strip(" ") for t in transcript_pieces])
        metadata = {"source": self.video_id}
        return [Document(page_content=transcript, metadata=metadata)]
