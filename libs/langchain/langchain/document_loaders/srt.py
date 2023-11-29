"""Loader for .srt (subtitle) files."""
from typing import List, Optional

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader
from langchain.utils.transcripts import chunk_transcripts, format_pysrt


class SRTLoader(BaseLoader):
    """Load `.srt` (subtitle) files."""

    def __init__(self, file_path: str, duration: Optional[int] = None):
        """Initialize with a file path."""
        try:
            import pysrt  # noqa:F401
        except ImportError:
            raise ImportError(
                "package `pysrt` not found, please install it with `pip install pysrt`"
            )
        self.file_path = file_path
        self.duration = duration

    def load(self) -> List[Document]:
        """Load using pysrt file."""
        import pysrt

        parsed_info = pysrt.open(self.file_path)
        metadata = {"source": self.file_path}
        if self.duration == None:
            text = " ".join([t.text for t in parsed_info])
            return [Document(page_content=text, metadata=metadata)]
        else:
            transcript_pieces = format_pysrt(parsed_info)
            transcript_pieces = chunk_transcripts(
                transcript_pieces, duration=self.duration
            )
            docs = []
            for t in transcript_pieces:
                dct = {
                    **metadata,
                    **{
                        "TimeStamp": t["start"],
                        "duration": t["duration"],
                    },
                }
                doc = Document(
                    page_content=t["text"],
                    metadata=dct,
                )
                docs += [doc]
            return docs
