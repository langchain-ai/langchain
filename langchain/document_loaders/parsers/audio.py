
from typing import Iterator
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document

class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse  audio files using audio-to-text transcription with OpenAI Whisper model."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import os
        import openai

        with blob as audio_file_path:
            audio_file = open(audio_file_path , 'rb')
            fpath , fname = os.path.split(audio_file_path)
            transcript = openai.Audio.transcribe("whisper-1",audio_file)
            yield Document(page_content=transcript.text,metadata={"source":fname})
