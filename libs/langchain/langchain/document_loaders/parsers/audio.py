import time
from typing import Iterator, Optional

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.
    Audio transcription is with OpenAI Whisper model."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io

        try:
            import openai
        except ImportError:
            raise ValueError(
                "openai package not found, please install it with "
                "`pip install openai`"
            )
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ValueError(
                "pydub package not found, please install it with " "`pip install pydub`"
            )

        # Set the API key if provided
        if self.api_key:
            openai.api_key = self.api_key

        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)

        # Define the duration of each chunk in minutes
        # Need to meet 25MB size limit for Whisper API
        chunk_duration = 20
        chunk_duration_ms = chunk_duration * 60 * 1000

        # Split the audio into chunk_duration_ms chunks
        for split_number, i in enumerate(range(0, len(audio), chunk_duration_ms)):
            # Audio chunk
            chunk = audio[i : i + chunk_duration_ms]
            file_obj = io.BytesIO(chunk.export(format="mp3").read())
            if blob.source is not None:
                file_obj.name = blob.source + f"_part_{split_number}.mp3"
            else:
                file_obj.name = f"part_{split_number}.mp3"

            # Transcribe
            print(f"Transcribing part {split_number+1}!")
            attempts = 0
            while attempts < 3:
                try:
                    transcript = openai.Audio.transcribe("whisper-1", file_obj)
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed. Exception: {str(e)}")
                    time.sleep(5)
            else:
                print("Failed to transcribe after 3 attempts.")
                continue

            yield Document(
                page_content=transcript.text,
                metadata={"source": blob.source, "chunk": split_number},
            )
