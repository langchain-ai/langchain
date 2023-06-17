"""Loader that loads YouTube transcript."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse

from pydantic import root_validator
from pydantic.dataclasses import dataclass

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]


@dataclass
class GoogleApiClient:
    """A Generic Google Api Client.

    To use, you should have the ``google_auth_oauthlib,youtube_transcript_api,google``
    python package installed.
    As the google api expects credentials you need to set up a google account and
    register your Service. "https://developers.google.com/docs/api/quickstart/python"



    Example:
        .. code-block:: python

            from langchain.document_loaders import GoogleApiClient
            google_api_client = GoogleApiClient(
                service_account_path=Path("path_to_your_sec_file.json")
            )

    """

    credentials_path: Path = Path.home() / ".credentials" / "credentials.json"
    service_account_path: Path = Path.home() / ".credentials" / "credentials.json"
    token_path: Path = Path.home() / ".credentials" / "token.json"

    def __post_init__(self) -> None:
        self.creds = self._load_credentials()

    @root_validator
    def validate_channel_or_videoIds_is_set(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
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


ALLOWED_SCHEMAS = {"http", "https"}
ALLOWED_NETLOCK = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
    "vid.plus",
}


def _parse_video_id(url: str) -> Optional[str]:
    """Parse a youtube url and return the video id if valid, otherwise None."""
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ALLOWED_SCHEMAS:
        return None

    if parsed_url.netloc not in ALLOWED_NETLOCK:
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


class YoutubeLoader(BaseLoader):
    """Loader that loads Youtube transcripts."""

    def __init__(
        self,
        video_id: str,
        add_video_info: bool = False,
        language: Union[str, Sequence[str]] = "en",
        translation: str = "en",
        continue_on_failure: bool = False,
    ):
        """Initialize with YouTube video ID."""
        self.video_id = video_id
        self.add_video_info = add_video_info
        self.language = language
        if isinstance(language, str):
            self.language = [language]
        else:
            self.language = language
        self.translation = translation
        self.continue_on_failure = continue_on_failure

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """Extract video id from common YT urls."""
        video_id = _parse_video_id(youtube_url)
        if not video_id:
            raise ValueError(
                f"Could not determine the video ID for the URL {youtube_url}"
            )
        return video_id

    @classmethod
    def from_youtube_url(cls, youtube_url: str, **kwargs: Any) -> YoutubeLoader:
        """Given youtube URL, load video."""
        video_id = cls.extract_video_id(youtube_url)
        return cls(video_id, **kwargs)

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from youtube_transcript_api import (
                NoTranscriptFound,
                TranscriptsDisabled,
                YouTubeTranscriptApi,
            )
        except ImportError:
            raise ImportError(
                "Could not import youtube_transcript_api python package. "
                "Please install it with `pip install youtube-transcript-api`."
            )

        metadata = {"source": self.video_id}

        if self.add_video_info:
            # Get more video meta info
            # Such as title, description, thumbnail url, publish_date
            video_info = self._get_video_info()
            metadata.update(video_info)

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
        except TranscriptsDisabled:
            return []

        try:
            transcript = transcript_list.find_transcript(self.language)
        except NoTranscriptFound:
            en_transcript = transcript_list.find_transcript(["en"])
            transcript = en_transcript.translate(self.translation)

        transcript_pieces = transcript.fetch()

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
                "Please install it with `pip install pytube`."
            )
        yt = YouTube(f"https://www.youtube.com/watch?v={self.video_id}")
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
    """Loader that loads all Videos from a Channel

    To use, you should have the ``googleapiclient,youtube_transcript_api``
    python package installed.
    As the service needs a google_api_client, you first have to initialize
    the GoogleApiClient.

    Additionally you have to either provide a channel name or a list of videoids
    "https://developers.google.com/docs/api/quickstart/python"



    Example:
        .. code-block:: python

            from langchain.document_loaders import GoogleApiClient
            from langchain.document_loaders import GoogleApiYoutubeLoader
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

    @root_validator
    def validate_channel_or_videoIds_is_set(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        request = self.youtube_client.search().list(
            part="id,snippet",
            channelId=channel_id,
            maxResults=50,  # adjust this value to retrieve more or fewer videos
        )
        video_ids = []
        while request is not None:
            response = request.execute()

            # Add each video ID to the list
            for item in response["items"]:
                if not item["id"].get("videoId"):
                    continue
                meta_data = {"videoId": item["id"]["videoId"]}
                if self.add_video_info:
                    item["snippet"].pop("thumbnails")
                    meta_data.update(item["snippet"])
                try:
                    page_content = self._get_transcripe_for_video_id(
                        item["id"]["videoId"]
                    )
                    video_ids.append(
                        Document(
                            page_content=page_content,
                            metadata=meta_data,
                        )
                    )
                except (TranscriptsDisabled, NoTranscriptFound) as e:
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
