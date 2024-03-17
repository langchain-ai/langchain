from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Iterator, Optional

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    import assemblyai


class TranscriptFormat(Enum):
    """Transcript format to use for the document loader."""

    TEXT = "text"
    """One document with the transcription text"""
    SENTENCES = "sentences"
    """Multiple documents, splits the transcription by each sentence"""
    PARAGRAPHS = "paragraphs"
    """Multiple documents, splits the transcription by each paragraph"""
    SUBTITLES_SRT = "subtitles_srt"
    """One document with the transcript exported in SRT subtitles format"""
    SUBTITLES_VTT = "subtitles_vtt"
    """One document with the transcript exported in VTT subtitles format"""


class AssemblyAIAudioTranscriptLoader(BaseLoader):
    """
    Loader for AssemblyAI audio transcripts.

    It uses the AssemblyAI API to transcribe audio files
    and loads the transcribed text into one or more Documents,
    depending on the specified format.

    To use, you should have the ``assemblyai`` python package installed, and the
    environment variable ``ASSEMBLYAI_API_KEY`` set with your API key.
    Alternatively, the API key can also be passed as an argument.

    Audio files can be specified via an URL or a local file path.
    """

    def __init__(
        self,
        file_path: str,
        *,
        transcript_format: TranscriptFormat = TranscriptFormat.TEXT,
        config: Optional[assemblyai.TranscriptionConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the AssemblyAI AudioTranscriptLoader.

        Args:
            file_path: An URL or a local file path.
            transcript_format: Transcript format to use.
                See class ``TranscriptFormat`` for more info.
            config: Transcription options and features. If ``None`` is given,
                the Transcriber's default configuration will be used.
            api_key: AssemblyAI API key.
        """
        try:
            import assemblyai
        except ImportError:
            raise ImportError(
                "Could not import assemblyai python package. "
                "Please install it with `pip install assemblyai`."
            )
        if api_key is not None:
            assemblyai.settings.api_key = api_key

        self.file_path = file_path
        self.transcript_format = transcript_format
        self.transcriber = assemblyai.Transcriber(config=config)

    def lazy_load(self) -> Iterator[Document]:
        """Transcribes the audio file and loads the transcript into documents.

        It uses the AssemblyAI API to transcribe the audio file and blocks until
        the transcription is finished.
        """
        transcript = self.transcriber.transcribe(self.file_path)
        # This will raise a ValueError if no API key is set.

        if transcript.error:
            raise ValueError(f"Could not transcribe file: {transcript.error}")

        if self.transcript_format == TranscriptFormat.TEXT:
            yield Document(
                page_content=transcript.text, metadata=transcript.json_response
            )
        elif self.transcript_format == TranscriptFormat.SENTENCES:
            sentences = transcript.get_sentences()
            for s in sentences:
                yield Document(page_content=s.text, metadata=s.dict(exclude={"text"}))
        elif self.transcript_format == TranscriptFormat.PARAGRAPHS:
            paragraphs = transcript.get_paragraphs()
            for p in paragraphs:
                yield Document(page_content=p.text, metadata=p.dict(exclude={"text"}))
        elif self.transcript_format == TranscriptFormat.SUBTITLES_SRT:
            yield Document(page_content=transcript.export_subtitles_srt())
        elif self.transcript_format == TranscriptFormat.SUBTITLES_VTT:
            yield Document(page_content=transcript.export_subtitles_vtt())
        else:
            raise ValueError("Unknown transcript format.")


class AssemblyAIAudioLoaderById(BaseLoader):
    """
    Loader for AssemblyAI audio transcripts.

    It uses the AssemblyAI API to get an existing transcription
    and loads the transcribed text into one or more Documents,
    depending on the specified format.

    """

    def __init__(
        self, transcript_id: str, api_key: str, transcript_format: TranscriptFormat
    ):
        """
        Initializes the AssemblyAI AssemblyAIAudioLoaderById.

        Args:
            transcript_id: Id of an existing transcription.
            transcript_format: Transcript format to use.
                See class ``TranscriptFormat`` for more info.
            api_key: AssemblyAI API key.
        """

        self.api_key = api_key
        self.transcript_id = transcript_id
        self.transcript_format = transcript_format

    def lazy_load(self) -> Iterator[Document]:
        """Load data into Document objects."""
        HEADERS = {"authorization": self.api_key}

        if self.transcript_format == TranscriptFormat.TEXT:
            try:
                transcript_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{self.transcript_id}",
                    headers=HEADERS,
                )
                transcript_response.raise_for_status()
            except Exception as e:
                print(f"An error occurred: {e}")  # noqa: T201
                raise

            transcript = transcript_response.json()["text"]

            yield Document(page_content=transcript, metadata=transcript_response.json())
        elif self.transcript_format == TranscriptFormat.PARAGRAPHS:
            try:
                paragraphs_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{self.transcript_id}/paragraphs",
                    headers=HEADERS,
                )
                paragraphs_response.raise_for_status()
            except Exception as e:
                print(f"An error occurred: {e}")  # noqa: T201
                raise

            paragraphs = paragraphs_response.json()["paragraphs"]

            for p in paragraphs:
                yield Document(page_content=p["text"], metadata=p)

        elif self.transcript_format == TranscriptFormat.SENTENCES:
            try:
                sentences_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{self.transcript_id}/sentences",
                    headers=HEADERS,
                )
                sentences_response.raise_for_status()
            except Exception as e:
                print(f"An error occurred: {e}")  # noqa: T201
                raise

            sentences = sentences_response.json()["sentences"]

            for s in sentences:
                yield Document(page_content=s["text"], metadata=s)

        elif self.transcript_format == TranscriptFormat.SUBTITLES_SRT:
            try:
                srt_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{self.transcript_id}/srt",
                    headers=HEADERS,
                )
                srt_response.raise_for_status()
            except Exception as e:
                print(f"An error occurred: {e}")  # noqa: T201
                raise

            srt = srt_response.text

            yield Document(page_content=srt)

        elif self.transcript_format == TranscriptFormat.SUBTITLES_VTT:
            try:
                vtt_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{self.transcript_id}/vtt",
                    headers=HEADERS,
                )
                vtt_response.raise_for_status()
            except Exception as e:
                print(f"An error occurred: {e}")  # noqa: T201
                raise

            vtt = vtt_response.text

            yield Document(page_content=vtt)
        else:
            raise ValueError("Unknown transcript format.")
