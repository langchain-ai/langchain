"""Loader that loads audio files."""
from typing import List

from langchain.audio_models.base import AudioBase
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class AudioLoader(BaseLoader):
    """Loader that loads audio files and converts
    to text using an audio model."""

    def __init__(
        self, audio_model: AudioBase, file_path: str, task: str = "transcribe"
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.file_path = file_path
        self.task = task

    def load(self) -> List[Document]:
        """Load audio file then convert to text. \
        Select among 'transcribe' and 'translate' tasks"""
        raw_content = self.audio_model.transcript(self.file_path, self.task)
        content = raw_content.strip()
        metadata = {"source": self.file_path, "task": self.task}
        return [Document(page_content=content, metadata=metadata)]
