"""Document loader for Indian language text files with encoding support."""
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


class IndicTextLoader(BaseLoader):
    """
    Loader for Indian language text documents.
    
    Supports languages: Hindi, Marathi, Bengali, Tamil,
    Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia, Urdu, Assamese.
    
    Handles UTF-8 encoding properly for Devanagari and 
    other Indic scripts.
    
    Example:
        .. code-block:: python
        
            from langchain_community.document_loaders import IndicTextLoader
            
            loader = IndicTextLoader(
                file_path="hindi_document.txt",
                language="hindi",
                encoding="utf-8"
            )
            docs = loader.load()
    """

    SUPPORTED_LANGUAGES = {
        "hindi": "hi",
        "marathi": "mr",
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "gujarati": "gu",
        "punjabi": "pa",
        "odia": "or",
        "urdu": "ur",
        "assamese": "as"
    }

    def __init__(
        self,
        file_path: str,
        language: str = "hindi",
        encoding: str = "utf-8",
        metadata: Optional[dict] = None
    ):
        """
        Initialize the IndicTextLoader.

        Args:
            file_path: Path to the text file.
            language: Indian language name (default: hindi).
            encoding: File encoding (default: utf-8).
            metadata: Optional additional metadata.
        """
        self.file_path = Path(file_path)
        self.language = language.lower()
        self.encoding = encoding
        self.extra_metadata = metadata or {}

        if self.language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported: {list(self.SUPPORTED_LANGUAGES.keys())}"
            )

    def load(self) -> List[Document]:
        """Load and return documents from the file."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )

        try:
            with open(
                self.file_path, 
                "r", 
                encoding=self.encoding,
                errors="replace"
            ) as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback for legacy encodings
            with open(
                self.file_path, 
                "r", 
                encoding="latin-1"
            ) as f:
                text = f.read()

        metadata = {
            "source": str(self.file_path),
            "language": self.language,
            "language_code": self.SUPPORTED_LANGUAGES[self.language],
            "encoding": self.encoding,
            "file_size_bytes": self.file_path.stat().st_size,
            **self.extra_metadata
        }

        return [Document(page_content=text, metadata=metadata)]

    def lazy_load(self):
        """Lazy load documents."""
        yield from self.load()
