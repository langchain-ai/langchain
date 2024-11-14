"""Loads YouTube transcript."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse
from xml.etree.ElementTree import ParseError  # OK: trusted-source

from langchain_core.documents import Document
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]


@dataclass
class GoogleApiClient:
    """Generic Google API Client.

    To use, you should have the ``google_auth_oauthlib,youtube_transcript_api,google``
    python package installed.
    As the google api expects credentials you need to set up a google account and
    register your Service. "https://developers.google.com/docs/api/quickstart/python"

    *Security Note*: Note that parsing of the transcripts relies on the standard
        xml library but the input is viewed as trusted in this case.


    Example:
        .. code-block:: python

            from langchain_community.document_loaders import GoogleApiClient
            google_api_client = GoogleApiClient(
                service_account_path=Path("path_to_your_sec_file.json")
            )

    """

    credentials_path: Path = Path.home() / ".credentials" / "credentials.json"
    service_account_path: Path = Path.home() / ".credentials" / "credentials.json"
    token_path: Path = Path.home() / ".credentials" / "token.json"

    def __post_init__(self) -> None:
        self.creds = self._load_credentials()

    @model_validator(mode="before")
    @classmethod
    def validate_channel_or_videoIds_is_set(cls, values: Dict[str, Any]) -> Any:
        """Validate that either folder_id or document_ids is set, but not both."""

        if not values.get("credentials_path") and not values.get(
            "service_account_path"
        ):
            raise ValueError("Must specify either channel_name or video_ids")
        return values

    def _load_credentials(self) -> Any:
        """Load credentials."""
        # Adapted from https://developers.google.com/drive/api/v3/quickstart/python
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401
        except ImportError:
            raise ImportError(
                "You must run"
                "`pip install --upgrade "
                "google-api-python-client google-auth-httplib2 "
                "google-auth-oauthlib "
                "youtube-transcript-api` "
                "to use the Google Drive loader"
            )

        creds = None
        if self.service_account_path.exists():
            return service_account.Credentials.from_service_account_file(
                str(self.service_account_path)
            )
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        return creds


ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_NETLOCS = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
    "vid.plus",
}


def _parse_video_id(url: str) -> Optional[str]:
    """Parse a YouTube URL and return the video ID if valid, otherwise None."""
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        return None

    if parsed_url.netloc not in ALLOWED_NETLOCS:
        return None

    path = parsed_url.path

    if path.endswith("/watch"):
        query = parsed_url.query
        parsed_query = parse_qs(query)
        if "v" in parsed_query:
            ids = parsed_query["v"]
            video_id = ids if isinstance(ids, str) else ids[0]
        else:
            return None
    else:
        path = parsed_url.path.lstrip("/")
        video_id = path.split("/")[-1]

    if len(video_id) != 11:  # Video IDs are 11 characters long
        return None

    return video_id


class TranscriptFormat(Enum):
    """Output formats of transcripts from `YoutubeLoader`."""

    TEXT = "text"
    LINES = "lines"
    CHUNKS = "chunks"


class YoutubeLoader(BaseLoader):
    """Load `YouTube` video transcripts."""

    def __init__(
        self,
        video_id: str,
        add_video_info: bool = False,
        language: Union[str, Sequence[str]] = "en",
        translation: Optional[str] = None,
        transcript_format: TranscriptFormat = TranscriptFormat.TEXT,
        continue_on_failure: bool = False,
        chunk_size_seconds: int = 120,
    ):
        """Initialize with YouTube video ID."""
        self.video_id = video_id
        self._metadata = {"source": video_id}
        self.add_video_info = add_video_info
        self.language = language
        if isinstance(language, str):
            self.language = [language]
        else:
            self.language = language
        self.translation = translation
        self.transcript_format = transcript_format
        self.continue_on_failure = continue_on_failure
        self.chunk_size_seconds = chunk_size_seconds

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """Extract video ID from common YouTube URLs."""
        video_id = _parse_video_id(youtube_url)
        if not video_id:
            raise ValueError(
                f'Could not determine the video ID for the URL "{youtube_url}".'
            )
        return video_id

    @classmethod
    def from_youtube_url(cls, youtube_url: str, **kwargs: Any) -> YoutubeLoader:
        """Given a YouTube URL, construct a loader.
        See `YoutubeLoader()` constructor for a list of keyword arguments.
        """
        video_id = cls.extract_video_id(youtube_url)
        return cls(video_id, **kwargs)

    def _make_chunk_document(
        self, chunk_pieces: List[Dict], chunk_start_seconds: int
    ) -> Document:
        """Create Document from chunk of transcript pieces."""
        m, s = divmod(chunk_start_seconds, 60)
        h, m = divmod(m, 60)
        return Document(
            page_content=" ".join(
                map(lambda chunk_piece: chunk_piece["text"].strip(" "), chunk_pieces)
            ),
            metadata={
                **self._metadata,
                "start_seconds": chunk_start_seconds,
                "start_timestamp": f"{h:02d}:{m:02d}:{s:02d}",
                "source":
                # replace video ID with URL to start time
                f"https://www.youtube.com/watch?v={self.video_id}"
                f"&t={chunk_start_seconds}s",
            },
        )

    def _get_transcript_chunks(
        self, transcript_pieces: List[Dict]
    ) -> Generator[Document, None, None]:
        chunk_pieces: List[Dict[str, Any]] = []
        chunk_start_seconds = 0
        chunk_time_limit = self.chunk_size_seconds
        for transcript_piece in transcript_pieces:
            piece_end = transcript_piece["start"] + transcript_piece["duration"]
            if piece_end > chunk_time_limit:
                if chunk_pieces:
                    yield self._make_chunk_document(chunk_pieces, chunk_start_seconds)
                chunk_pieces = []
                chunk_start_seconds = chunk_time_limit
                chunk_time_limit += self.chunk_size_seconds

            chunk_pieces.append(transcript_piece)

        if len(chunk_pieces) > 0:
            yield self._make_chunk_document(chunk_pieces, chunk_start_seconds)

    def load(self) -> List[Document]:
        """Load YouTube transcripts into `Document` objects."""
        try:
            from youtube_transcript_api import (
                NoTranscriptFound,
                TranscriptsDisabled,
                YouTubeTranscriptApi,
            )
        except ImportError:
            raise ImportError(
                'Could not import "youtube_transcript_api" Python package. '
                "Please install it with `pip install youtube-transcript-api`."
            )

        if self.add_video_info:
            # Get more video meta info
            # Such as title, description, thumbnail url, publish_date
            video_info = self._get_video_info()
            self._metadata.update(video_info)

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
        except TranscriptsDisabled:
            return []

        try:
            transcript = transcript_list.find_transcript(self.language)
        except NoTranscriptFound:
            transcript = transcript_list.find_transcript(["en"])

        if self.translation is not None:
            transcript = transcript.translate(self.translation)

        transcript_pieces: List[Dict[str, Any]] = transcript.fetch()

        if self.transcript_format == TranscriptFormat.TEXT:
            transcript = " ".join(
                map(
                    lambda transcript_piece: transcript_piece["text"].strip(" "),
                    transcript_pieces,
                )
            )
            return [Document(page_content=transcript, metadata=self._metadata)]
        elif self.transcript_format == TranscriptFormat.LINES:
            return list(
                map(
                    lambda transcript_piece: Document(
                        page_content=transcript_piece["text"].strip(" "),
                        metadata=dict(
                            filter(
                                lambda item: item[0] != "text", transcript_piece.items()
                            )
                        ),
                    ),
                    transcript_pieces,
                )
            )
        elif self.transcript_format == TranscriptFormat.CHUNKS:
            return list(self._get_transcript_chunks(transcript_pieces))

        else:
            raise ValueError("Unknown transcript format.")

    def _get_video_info(self) -> Dict:
        """Get important video information.

        Components include:
            - title
            - description
            - thumbnail URL,
            - publish_date
            - channel author
            - and more.
        """
        try:
            from pytube import YouTube

        except ImportError:
            raise ImportError(
                'Could not import "pytube" Python package. '
                "Please install it with `pip install pytube`."
            )
        yt_url = f"https://www.youtube.com/watch?v={self.video_id}"    
        yt = YouTube(
             yt_url,
             use_oauth=True,
             allow_oauth_cache=True
            
            )
        video_info = {
            "title": yt.title or "Unknown",
            "description": yt.description or "Unknown",
            "view_count": yt.views or 0,
            "thumbnail_url": yt.thumbnail_url or "Unknown",
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S")
            if yt.publish_date
            else "Unknown",
            "length": yt.length or 0,
            "author": yt.author or "Unknown",
        }
        return video_info


@dataclass
class GoogleApiYoutubeLoader(BaseLoader):
    """Load all Videos from a `YouTube` Channel.

    To use, you should have the ``googleapiclient,youtube_transcript_api``
    python package installed.
    As the service needs a google_api_client, you first have to initialize
    the GoogleApiClient.

    Additionally you have to either provide a channel name or a list of videoids
    "https://developers.google.com/docs/api/quickstart/python"



    Example:
        .. code-block:: python

            from langchain_community.document_loaders import GoogleApiClient
            from langchain_community.document_loaders import GoogleApiYoutubeLoader
            google_api_client = GoogleApiClient(
                service_account_path=Path("path_to_your_sec_file.json")
            )
            loader = GoogleApiYoutubeLoader(
                google_api_client=google_api_client,
                channel_name = "CodeAesthetic"
            )
            load.load()

    """

    google_api_client: GoogleApiClient
    channel_name: Optional[str] = None
    video_ids: Optional[List[str]] = None
    add_video_info: bool = True
    captions_language: str = "en"
    continue_on_failure: bool = False

    def __post_init__(self) -> None:
        self.youtube_client = self._build_youtube_client(self.google_api_client.creds)

    def _build_youtube_client(self, creds: Any) -> Any:
        try:
            from googleapiclient.discovery import build
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401
        except ImportError:
            raise ImportError(
                "You must run"
                "`pip install --upgrade "
                "google-api-python-client google-auth-httplib2 "
                "google-auth-oauthlib "
                "youtube-transcript-api` "
                "to use the Google Drive loader"
            )

        return build("youtube", "v3", credentials=creds)

    @model_validator(mode="before")
    @classmethod
    def validate_channel_or_videoIds_is_set(cls, values: Dict[str, Any]) -> Any:
        """Validate that either folder_id or document_ids is set, but not both."""
        if not values.get("channel_name") and not values.get("video_ids"):
            raise ValueError("Must specify either channel_name or video_ids")
        return values

    def _get_transcripe_for_video_id(self, video_id: str) -> str:
        from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript([self.captions_language])
        except NoTranscriptFound:
            for available_transcript in transcript_list:
                transcript = available_transcript.translate(self.captions_language)
                continue

        transcript_pieces = transcript.fetch()
        return " ".join([t["text"].strip(" ") for t in transcript_pieces])

    def _get_document_for_video_id(self, video_id: str, **kwargs: Any) -> Document:
        captions = self._get_transcripe_for_video_id(video_id)
        video_response = (
            self.youtube_client.videos()
            .list(
                part="id,snippet",
                id=video_id,
            )
            .execute()
        )
        return Document(
            page_content=captions,
            metadata=video_response.get("items")[0],
        )

    def _get_channel_id(self, channel_name: str) -> str:
        request = self.youtube_client.search().list(
            part="id",
            q=channel_name,
            type="channel",
            maxResults=1,  # we only need one result since channel names are unique
        )
        response = request.execute()
        channel_id = response["items"][0]["id"]["channelId"]
        return channel_id

    def _get_uploads_playlist_id(self, channel_id: str) -> str:
        request = self.youtube_client.channels().list(
            part="contentDetails",
            id=channel_id,
        )
        response = request.execute()
        return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    def _get_document_for_channel(self, channel: str, **kwargs: Any) -> List[Document]:
        try:
            from youtube_transcript_api import (
                NoTranscriptFound,
                TranscriptsDisabled,
            )
        except ImportError:
            raise ImportError(
                "You must run"
                "`pip install --upgrade "
                "youtube-transcript-api` "
                "to use the youtube loader"
            )

        channel_id = self._get_channel_id(channel)
        uploads_playlist_id = self._get_uploads_playlist_id(channel_id)
        request = self.youtube_client.playlistItems().list(
            part="id,snippet",
            playlistId=uploads_playlist_id,
            maxResults=50,
        )
        video_ids = []
        while request is not None:
            response = request.execute()

            # Add each video ID to the list
            for item in response["items"]:
                video_id = item["snippet"]["resourceId"]["videoId"]
                meta_data = {"videoId": video_id}
                if self.add_video_info:
                    item["snippet"].pop("thumbnails")
                    meta_data.update(item["snippet"])
                try:
                    page_content = self._get_transcripe_for_video_id(video_id)
                    video_ids.append(
                        Document(
                            page_content=page_content,
                            metadata=meta_data,
                        )
                    )
                except (TranscriptsDisabled, NoTranscriptFound, ParseError) as e:
                    if self.continue_on_failure:
                        logger.error(
                            "Error fetching transscript "
                            + f" {item['id']['videoId']}, exception: {e}"
                        )
                    else:
                        raise e
                    pass
            request = self.youtube_client.search().list_next(request, response)

        return video_ids

    def load(self) -> List[Document]:
        """Load documents."""
        document_list = []
        if self.channel_name:
            document_list.extend(self._get_document_for_channel(self.channel_name))
        elif self.video_ids:
            document_list.extend(
                [
                    self._get_document_for_video_id(video_id)
                    for video_id in self.video_ids
                ]
            )
        else:
            raise ValueError("Must specify either channel_name or video_ids")
        return document_list
