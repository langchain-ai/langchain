from typing import Iterator

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.
    Audio transcription is with OpenAI Whisper model."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import openai

        with blob.as_bytes_io() as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
            yield Document(
                page_content=transcript.text, metadata={"source": blob.source}
            )
