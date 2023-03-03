"""Loader that loads MP3 files."""
from typing import List

from langchain.audio_models.base import AudioBase
from langchain.docstore.document import Document
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class MP3Loader(UnstructuredFileLoader):
    """Loader that loads MP3 files and converts to text using an audio model."""

    audio_model: AudioBase

    def load(self) -> List[Document]:
        """Load MP3 file then convert to text."""
        return [
            Document(page_content=self.audio_model.transcript(self.file_path).strip())
        ]
