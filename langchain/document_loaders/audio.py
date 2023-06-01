import os
import openai
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class AudioFileLoader(BaseLoader):
    """Load Audio file by first transcribing with OpenAI Whisper model."""

    def __init__(self, audio_file_path: str = "text"):
        """Initialize with audio file path."""
        self.audio_file_path = audio_file_path

    def load(self) -> Document:
        """Transcribe w/ OpenAI Whisper. Note: 25MB file size limit."""
        audio_file = open(self.audio_file_path , "rb")
        fpath , fname = os.path.split(self.audio_file_path)
        transcript = openai.Audio.transcribe("whisper-1",audio_file)
        result = Document(page_content=transcript.text,metadata={"file_name":fname})
        return result