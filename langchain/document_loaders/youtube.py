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

    def load(self,  add_video_infor: bool=False) -> List[Document]:
        """Load documents."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "Could not import youtube_transcript_api python package. "
                "Please it install it with `pip install youtube-transcript-api`."
            )
        
        metadata = {"source": self.video_id}

        if add_video_infor:
            # Get more video meta info, such as title, description, thumbnail url, publish_date， 
            video_infor = self._get_video_infor()
            metadata.update(video_infor)

        transcript_pieces = YouTubeTranscriptApi.get_transcript(self.video_id)
        transcript = " ".join([t["text"].strip(" ") for t in transcript_pieces])
        
        return [Document(page_content=transcript, metadata=metadata)]

    def _get_video_infor(self):
        """Get important video information: title, description, thumbnail url, publish_date, channel_author and more."""
        try:
            from pytube import YouTube

        except ImportError:
            raise ImportError(
                "Could not import pytube python package. "
                "Please it install it with `pip install pytube`."
            )
        yt = YouTube(f"https://www.youtube.com/watch?v={self.video_id}")
        video_info = {
            'title': yt.title,
            'description': yt.description,
            'view_count': yt.views,
            'thumbnail_url': yt.thumbnail_url,
            'publish_date': yt.publish_date,
            'length': yt.length,
            'author': yt.author,
        }
        return video_info