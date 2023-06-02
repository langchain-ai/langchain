from typing import Iterator

from langchain.document_loaders.base import BaseBlobSplitter
from langchain.document_loaders.blob_loaders import Blob


class AudioSplitter(BaseBlobSplitter):

    """Dump YouTube url as mp3 file."""

    # TODO: Output should be a list of the blob paths?
    def lazy_split(self, blob: Blob) -> Iterator[Blob]:
        """Lazily split the blob."""

        from pydub import AudioSegment

        # TODO: Determine best way to pass this
        output_file_path = "path_to_files"

        video = AudioSegment.from_file(blob)

        # Define the duration of each chunk in minutes
        chunk_duration = 20

        # Calculate the chunk duration in milliseconds
        chunk_duration_ms = chunk_duration * 60 * 1000

        # Split the audio into chunk_duration-minute chunks
        for i in range(0, len(video), chunk_duration_ms):
            chunk = video[i : i + chunk_duration_ms]
            fpath = output_file_path + f"audio_chunk_{i // chunk_duration_ms}.mp3"
            chunk.export(fpath, format="mp3")
            yield Blob.from_path(fpath)
