from typing import Iterator

from langchain.document_loaders.base import BaseBlobSplitter
from langchain.document_loaders.blob_loaders import Blob


class AudioSplitter(BaseBlobSplitter):

    """Dump YouTube url as mp3 file."""

    def lazy_split(self, blob: Blob) -> Iterator[Blob]:
        
        """Lazily split the blob."""

        from pydub import AudioSegment

        # Blob.data is BytesIO object to store audio file in memory
        audio = AudioSegment.from_file(blob.path)

        # Define the duration of each chunk in minutes
        chunk_duration = 20

        # Calculate the chunk duration in milliseconds
        chunk_duration_ms = chunk_duration * 60 * 1000

        # Split the audio into chunk_duration_ms chunks and return blobs
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i : i + chunk_duration_ms]
            yield Blob.from_data(chunk.export(format="mp3").read())
    
