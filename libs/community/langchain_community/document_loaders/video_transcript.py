"""
Setup Instructions for the video loaders

Requirements:
- Python 3.8 or higher

ffmpeg Setup:
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using direct download:
Download from https://ffmpeg.org/download.html and add the executable to your PATH.

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
"""

import logging
import os
import shutil
import subprocess
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _format_start_time(seconds: float) -> str:
    """
    Formats the given number of seconds
    into a human-readable time string.

    Parameters:
        seconds (float): The number of seconds to format.

    Returns:
        str: The formatted time string in the format "HH:MM:SS"
        if the number of hours is greater than 0, or "MM:SS"
        if the number of minutes is greater than 0, or "0:SS"
        if neither hours nor minutes are present.
    """
    try:
        seconds = int(round(seconds))
    except TypeError:
        logger.exception("invalid number of seconds {seconds}.")
        return ""
    except Exception as e:
        logger.exception(f"failed to convert {seconds}: {str(e)}.")
        return ""

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"
    elif minutes > 0:
        return f"{minutes:02}:{remaining_seconds:02}"
    else:
        return f"0:{remaining_seconds:02}"


def _convert_video_to_ogg(input_path: str) -> str | None:
    """
    Converts the input video file to an OGG format using FFmpeg.

    Args:
        input_path (str): The path to the input video file.
    
    Returns:
        str: The path to the converted OGG file, or None if the conversion fails.

    Raises:
        subprocess.CalledProcessError: If there is an error 
        during the conversion process.
        ImportError: If ffmpeg is not found in the system's PATH.

    This function uses FFmpeg to convert the input video 
    file to an OGG format. It takes the input video file path as 
    a parameter and returns the path to the converted OGG file. 
    If the conversion fails, it prints an error 
    message and returns None.

    The FFmpeg command used for the conversion is as follows:

    ```bash
    ffmpeg -i input_path -vn -map_metadata -1 \
        -ac 1 -c:a libopus -b:a 12k \
        -application voip <output_path>
    ```

    Note: This function assumes that ffmpeg is 
    installed on the system and is available in the system's PATH.
    """

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise ImportError("ffmpeg is not found in the system's PATH.")

    logger.info(f"Starting video conversion for: {input_path}")

    output_path = input_path.rsplit(".", 1)[0] + ".ogg"
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "12k",
        "-application",
        "voip",
        output_path,
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"Video successfully converted to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during conversion of {input_path}: {e}")
        return None


class AzureWhisperVideoSegmentLoader(BaseLoader):
    """A document loader that processes video files, converts them to .ogg,
    and transcribes them using Azure OpenAI's API."""

    def __init__(
        self,
        video_path: str,
        deployment_id: str,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
    ) -> None:
        """Initialize the loader with a video path and Azure API details.

        Args:
            video_path: The path to the video file to load.
            deployment_id: The Azure deployment ID for the Whisper model.
            api_key: Azure API key for authentication.
            api_version: API version.
            azure_endpoint: Azure API endpoint.

        Raises:
            ImportError: If the OpenAI library is not installed.
        """
        try:
            from openai import AzureOpenAI
        except ImportError:
            logger.exception(
                "OpenAI library not found, please install it with `pip install openai`"
            )
            raise ImportError(
                "OpenAI library not found, please install it with `pip install openai`"
            )

        self.video_path = video_path
        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.deployment_id = deployment_id
        logger.info(
            f"Using Azure OpenAI Whisper model with "
            f"'{deployment_id}' for video path: {video_path}"
        )

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that processes video segments and yields Document objects."""
        logger.info(f"Started processing {self.video_path} through AzureOpenAI")

        converted_path = _convert_video_to_ogg(self.video_path)
        if not converted_path:
            logger.error("Video conversion failed, transcription aborted.")
            raise RuntimeError("Video conversion failed, transcription aborted.")

        transcription_result = self.transcribe_video(converted_path)
        if not transcription_result:
            logger.error("Transcription failed, no documents will be yielded.")
            return

        for seg in transcription_result.segments:
            text = seg["text"]
            metadata = {
                "id": seg["id"],
                "seek": seg["seek"],
                "start": _format_start_time(seg["start"]),
                "end": _format_start_time(seg["end"]),
                "tokens": seg["tokens"],
                "temperature": seg["temperature"],
                "avg_logprob": seg["avg_logprob"],
                "compression_ratio": seg["compression_ratio"],
                "no_speech_prob": seg["no_speech_prob"],
                "source": self.video_path,
            }

            logger.debug(f"Yielding document with metadata: {metadata}")
            yield Document(page_content=text, metadata=metadata)

        logger.info(f"Completed processing all segments for video: {self.video_path}")

    def transcribe_video(self, video_file: str) -> dict:
        """
        A function that transcribes a video file using Azure OpenAI's API.

        Parameters:
            video_file (str): The path to the video file to transcribe.

        Returns:
            dict: The transcription result in verbose JSON
            format or an empty dictionary if there is an error.
        """
        logger.info(f"Starting transcription for video file: {video_file}")
        if not os.path.exists(video_file):
            logger.error(f"The video file was not found: {video_file}")
            return {}

        try:
            with open(video_file, "rb") as file:
                result = self.client.audio.transcriptions.create(
                    file=file,
                    model=self.deployment_id,
                    temperature=0.01,
                    language="en",
                    response_format="verbose_json",
                    timestamp_granularities="segment",
                )
            logger.info(f"Transcription successful for: {video_file}")
            return result
        except Exception:
            logger.exception("unexpected error during transcription of video file")
            return {}


class AzureWhisperVideoParagraphLoader(BaseLoader):
    """
    A document loader that processes video files,
    converts them to .ogg, and transcribes them
    into paragraphs with predefined sentence
    size using Azure OpenAI's API.
    """

    def __init__(
        self,
        video_path: str,
        deployment_id: str,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        paragraph_sentence_size: int,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError:
            logger.error("openai is not installed")
            raise ImportError("openai is not installed")

        self.video_path = video_path
        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.deployment_id = deployment_id
        self.paragraph_sentence_size = paragraph_sentence_size
        logger.info(
            f"Using Azure OpenAI Whisper model with '{deployment_id}' "
            f"for video path: {video_path}"
        )

    def lazy_load(self) -> Iterator[Document]:
        """
        A lazy loader that processes video segments
        and yields Document objects.
        """
        logger.info(f"Started processing {self.video_path} through AzureOpenAI")

        converted_path = _convert_video_to_ogg(self.video_path)
        if not converted_path:
            logger.error("Video conversion failed, transcription aborted.")
            raise RuntimeError("Video conversion failed, transcription aborted.")

        transcription_result = self.transcribe_video(converted_path)
        if not transcription_result:
            logger.error("Transcription failed, no documents will be yielded.")
            return

        data = [
            {
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "source": self.video_path,
                "temperature": seg["temperature"],
                "avg_logprob": seg["avg_logprob"],
            }
            for seg in transcription_result.segments
        ]

        paragraphs = self.build_paragraphs(data)
        for paragraph in paragraphs:
            yield Document(
                page_content=paragraph["paragraph"],
                metadata={
                    "start_time": paragraph["start_time"],
                    "end_time": paragraph["end_time"],
                    "source": paragraph["source"],
                    "temperature": paragraph["temperature"],
                    "avg_logprob": paragraph["avg_logprob"],
                },
            )

    def transcribe_video(self, video_file: str) -> dict:
        """
        A function that transcribes a video file
        using Azure OpenAI's API.

        Parameters:
            video_file (str): The path to the video file to transcribe.

        Returns:
            dict: The transcription result in verbose JSON
            format or an empty dictionary if there is an error.
        """
        logger.info(f"Starting transcription for video file: {video_file}")
        if not os.path.exists(video_file):
            logger.error(f"The video file was not found: {video_file}")
            return {}

        try:
            with open(video_file, "rb") as file:
                result = self.client.audio.transcriptions.create(
                    file=file,
                    model=self.deployment_id,
                    temperature=0.01,
                    language="en",
                    response_format="verbose_json",
                    timestamp_granularities="segment",
                )
            logger.info(f"Transcription successful for: {video_file}")
            return result
        except Exception:
            logger.exception("unexpected error during transcription of video file")
            return {}

    def build_paragraphs(self, data) -> list:
        """
        Build paragraphs from transcribed data.
        """
        paragraphs = []
        i = 0
        while i < len(data):
            current_paragraph = []
            while len(
                current_paragraph
            ) < self.paragraph_sentence_size or not current_paragraph[-1][
                "text"
            ].strip().endswith("."):
                current_paragraph.append(data[i])
                i += 1
                if i >= len(data):
                    break

            paragraph = " ".join(entry["text"].strip() for entry in current_paragraph)
            start_time = _format_start_time(current_paragraph[0]["start"])
            end_time = _format_start_time(current_paragraph[-1]["end"])
            source = current_paragraph[0]["source"]
            temperature = current_paragraph[0]["temperature"]
            avg_logprob = current_paragraph[0]["avg_logprob"]

            paragraphs.append(
                {
                    "paragraph": paragraph,
                    "start_time": start_time,
                    "end_time": end_time,
                    "source": source,
                    "temperature": temperature,
                    "avg_logprob": avg_logprob,
                }
            )
        return paragraphs


class OpenAIWhisperVideoSegmentLoader(BaseLoader):
    """A document loader that processes video files, converts them to .ogg,
    and transcribes them using OpenAI's API."""

    def __init__(self, video_path: str, api_key: str) -> None:
        """Initialize the loader with a video path and OpenAI API details.

        Args:
            video_path: The path to the video file to load.
            api_key: OpenAI API key for authentication.

        Raises:
            ImportError: If the OpenAI library is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error(
                "OpenAI library not found, please install it with `pip install openai`"
            )
            raise ImportError(
                "OpenAI library not found, please install it with `pip install openai`"
            )

        self.video_path = video_path
        self.client = OpenAI(api_key=api_key)
        logger.info(
            f"Using OpenAI Whisper model with OpenAI API for video path: {video_path}"
        )

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that processes video segments and yields Document objects."""
        logger.info(
            f"Started processing video with Whisper through OpenAI: {self.video_path}"
        )

        converted_path = _convert_video_to_ogg(self.video_path)
        if not converted_path:
            logger.error("Video conversion failed, transcription aborted.")
            raise RuntimeError("Video conversion failed, transcription aborted.")

        transcription_result = self.transcribe_video(converted_path)
        if not transcription_result:
            logger.error("Transcription failed, no documents will be yielded.")
            return

        for seg in transcription_result.segments:
            text = seg["text"]
            metadata = {
                "id": seg["id"],
                "seek": seg["seek"],
                "start": _format_start_time(seg["start"]),
                "end": _format_start_time(seg["end"]),
                "tokens": seg["tokens"],
                "temperature": seg["temperature"],
                "avg_logprob": seg["avg_logprob"],
                "compression_ratio": seg["compression_ratio"],
                "no_speech_prob": seg["no_speech_prob"],
                "source": self.video_path,
            }

            logger.debug(f"Yielding document with metadata: {metadata}")
            yield Document(page_content=text, metadata=metadata)

        logger.info(f"Completed processing all segments for video: {self.video_path}")

    def transcribe_video(self, video_file: str) -> dict:
        """
        A function that transcribes a video
        file using Azure OpenAI's API.

        Parameters:
            video_file (str): The path to the
            video file to transcribe.

        Returns:
            dict: The transcription result in verbose
            JSON format or an empty dictionary if
            there is an error.
        """
        logger.info(f"Starting transcription for video file {video_file}")
        if not os.path.exists(video_file):
            logger.error(f"The video file {video_file} was not found")
            return {}

        try:
            with open(video_file, "rb") as file:
                result = self.client.audio.transcriptions.create(
                    file=file,
                    model="whisper-1",
                    temperature=0.01,
                    language="en",
                    response_format="verbose_json",
                    timestamp_granularities="segment",
                )
            logger.info(f"Transcription generated successful for: {video_file}")
            return result
        except Exception:
            logger.exception("unexpected error during transcription of video file")
            return {}


class OpenAIWhisperVideoParagraphLoader(BaseLoader):
    """
    A document loader that processes video files,
    converts them to .ogg, and transcribes them
    into paragraphs with predefined sentence size
    using OpenAI's API.
    """

    def __init__(
        self, video_path: str, api_key: str, paragraph_sentence_size: int
    ) -> None:
        """Initialize the loader with a video path and OpenAI API details.

        Args:
            video_path: The path to the video file to load.
            api_key: OpenAI API key for authentication.

        Raises:
            ImportError: If the OpenAI library is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not found, please install it with `pip install openai`"
            )

        self.video_path = video_path
        self.client = OpenAI(api_key=api_key)
        self.paragraph_sentence_size = paragraph_sentence_size
        logger.info(
            f"Using OpenAI Whisper model with OpenAI API for video path: {video_path}"
        )

    def lazy_load(self) -> Iterator[Document]:
        """
        A lazy loader that processes video segments
        and yields Document objects.
        """
        logger.info(f"Started processing {self.video_path} through AzureOpenAI")

        converted_path = _convert_video_to_ogg(self.video_path)
        if not converted_path:
            logger.error("Video conversion failed, transcription aborted.")
            raise RuntimeError("Video conversion failed, transcription aborted.")

        transcription_result = self.transcribe_video(converted_path)
        if not transcription_result:
            logger.error("Transcription failed, no documents will be yielded.")
            return

        data = [
            {
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "source": self.video_path,
                "temperature": seg["temperature"],
                "avg_logprob": seg["avg_logprob"],
            }
            for seg in transcription_result.segments
        ]

        paragraphs = self.build_paragraphs(data)
        for paragraph in paragraphs:
            yield Document(
                page_content=paragraph["paragraph"],
                metadata={
                    "start_time": paragraph["start_time"],
                    "end_time": paragraph["end_time"],
                    "source": paragraph["source"],
                    "temperature": paragraph["temperature"],
                    "avg_logprob": paragraph["avg_logprob"],
                },
            )

    def transcribe_video(self, video_file: str) -> dict:
        """
        A function that transcribes a video
        file using Azure OpenAI's API.

        Parameters:
            video_file (str): The path to the
            video file to transcribe.

        Returns:
            dict: The transcription result in verbose
            JSON format or an empty dictionary if there is an error.
        """
        logger.info(f"Starting transcription for video file: {video_file}")
        if not os.path.exists(video_file):
            logger.error(f"The video file was not found: {video_file}")
            return {}

        try:
            with open(video_file, "rb") as file:
                result = self.client.audio.transcriptions.create(
                    file=file,
                    model="whisper-1",
                    temperature=0.01,
                    language="en",
                    response_format="verbose_json",
                    timestamp_granularities="segment",
                )
            logger.info(f"Transcription successful for: {video_file}")
            return result
        except Exception:
            logger.exception("unexpected error during transcription of video file")
            return {}

    def build_paragraphs(self, data) -> list:
        paragraphs = []
        i = 0
        while i < len(data):
            current_paragraph = []
            while len(
                current_paragraph
            ) < self.paragraph_sentence_size or not current_paragraph[-1][
                "text"
            ].strip().endswith("."):
                current_paragraph.append(data[i])
                i += 1
                if i >= len(data):
                    break

            paragraph = " ".join(entry["text"].strip() for entry in current_paragraph)
            start_time = _format_start_time(current_paragraph[0]["start"])
            end_time = _format_start_time(current_paragraph[-1]["end"])
            source = current_paragraph[0]["source"]
            temperature = current_paragraph[0]["temperature"]
            avg_logprob = current_paragraph[0]["avg_logprob"]

            paragraphs.append(
                {
                    "paragraph": paragraph,
                    "start_time": start_time,
                    "end_time": end_time,
                    "source": source,
                    "temperature": temperature,
                    "avg_logprob": avg_logprob,
                }
            )
        return paragraphs


class LocalWhisperVideoSegmentLoader(BaseLoader):
    """
    A document loader that processes video
    files and transcribes them using
    Whisper locally.
    """

    def __init__(self, video_path: str, model_name="small") -> None:
        """
        Initialize the loader with a video
        path and a Whisper model.

        Args:
            video_path: The path to the video file to load.
            model_name: The Whisper model name to use
            for transcription.Defaults to "small".

        Raises:
            ImportError: If the Whisper library is not installed.
        """
        global whisper
        try:
            import whisper
        except ImportError:
            raise ImportError("openai-whisper is not installed")

        self.video_path = video_path
        self.model = whisper.load_model(model_name)
        logger.info(
            f"Loaded Whisper model '{model_name}' locally for video path: {video_path}"
        )

    def lazy_load(self) -> Iterator[Document]:
        """
        A lazy loader that processes video
        segments and yields Document objects.

        This method transcribes a video and
        yields each segment as a Document, including
        start time and source file as metadata.
        """
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        try:
            logger.info(f"Starting transcription for {self.video_path}")
            result = self.model.transcribe(
                self.video_path, temperature=0.01, language="en"
            )
            for seg in result["segments"]:
                text = seg["text"]
                id = str(seg["id"])
                seek = str(seg["seek"])
                start = _format_start_time(seg["start"])
                end = _format_start_time(seg["end"])
                tokens = seg["tokens"]
                temperature = str(seg["temperature"])
                avg_logprob = str(seg["avg_logprob"])
                compression_ratio = str(seg["compression_ratio"])
                no_speech_prob = str(seg["no_speech_prob"])
                metadata = {
                    "id": id,
                    "seek": seek,
                    "start": start,
                    "end": end,
                    "tokens": tokens,
                    "temperature": temperature,
                    "avg_logprob": avg_logprob,
                    "compression_ratio": compression_ratio,
                    "no_speech_prob": no_speech_prob,
                    "source": self.video_path,
                }
                logger.debug(f"Yielding Document with metadata: {metadata}")
                yield Document(page_content=text, metadata=metadata)
        except Exception as e:
            raise Exception(f"An error occurred during transcription: {e}")


class LocalWhisperVideoParagraphLoader(BaseLoader):
    """
    A document loader that processes video files
    and transcribes them into paragraphs with
    predefined sentence size using Whisper.
    """

    def __init__(
        self, video_path: str, paragraph_sentence_size: int, model_name="small"
    ) -> None:
        """
        Initialize the loader with a video path,
        a Whisper model, and paragraph sentence size.

        Args:
            video_path (str): The path to the video file to load.
            paragraph_sentence_size (int): The number of sentences
            to typically include in a paragraph.

            model_name (str): The Whisper model name to use for
            transcription. Defaults to "small".

        Raises:
            ImportError: If the Whisper library is not installed.
        """
        global whisper
        try:
            import whisper
        except ImportError:
            raise ImportError("openai-whisper is not installed")

        self.video_path = video_path
        self.paragraph_sentence_size = paragraph_sentence_size
        self.model = whisper.load_model(model_name)
        logger.info(
            f"Loaded Whisper model '{model_name}' locally for video path: {video_path}"
        )

    def build_paragraphs(self, data) -> list:
        """Build paragraphs from transcribed data."""
        paragraphs = []
        i = 0
        while i < len(data):
            current_paragraph = []
            while len(
                current_paragraph
            ) < self.paragraph_sentence_size or not current_paragraph[-1][
                "text"
            ].strip().endswith("."):
                current_paragraph.append(data[i])
                i += 1
                if i >= len(data):
                    break

            paragraph = " ".join(entry["text"].strip() for entry in current_paragraph)
            metadata = {
                "start_time": _format_start_time(current_paragraph[0]["start"]),
                "end_time": _format_start_time(current_paragraph[-1]["end"]),
                "source": current_paragraph[0]["source"],
                "temperature": current_paragraph[0]["temperature"],
                "avg_logprob": current_paragraph[0]["avg_logprob"],
            }
            paragraphs.append({"paragraph": paragraph, "metadata": metadata})
        return paragraphs

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that processes video segments and yields Document objects."""
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        try:
            logger.info(f"Starting transcription for {self.video_path}")
            result = self.model.transcribe(
                self.video_path, temperature=0.01, language="en"
            )
            data = [
                {
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "source": self.video_path,
                    "temperature": seg["temperature"],
                    "avg_logprob": seg["avg_logprob"],
                }
                for seg in result["segments"]
            ]
            paragraphs = self.build_paragraphs(data)
            for paragraph in paragraphs:
                logger.debug(
                    f"Yielding Document with metadata: {paragraph['metadata']}"
                )
                yield Document(
                    page_content=paragraph["paragraph"], metadata=paragraph["metadata"]
                )
        except Exception as e:
            logger.exception(f"An error occurred during transcription: {e}")
            raise Exception(f"An error occurred during transcription: {e}")
