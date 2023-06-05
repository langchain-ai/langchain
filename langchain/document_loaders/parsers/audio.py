from typing import Iterator

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.
    Audio transcription is with OpenAI Whisper model."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io 
        import openai

        # Need if we pass an in-memory blob from AudioSplitter().lazy_split(video)
        file_obj = io.BytesIO(blob.data)
        file_obj.name = "input_split.mp3"

        # This fails when split is not read from disk (e.g., w/ GenericLoader)
        # OAI API is missing "f.name"
        with blob.as_bytes_io() as f:
            # *** This is a hack to unblock ***
            f = file_obj
            transcript = openai.Audio.transcribe("whisper-1", f)
            yield Document(
                page_content=transcript.text, metadata={"source": blob.source}
            )
