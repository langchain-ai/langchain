"""
This Document loader supports the following file extensions:
- .pdf (Portable Document Format)
- .txt (Plain Text)
- .docx (Microsoft Word Document)
- .pptx (Microsoft PowerPoint)
- .xlsx (Microsoft Excel)
- .rtf (Rich Text Format)
- .epub (EPUB eBook)
- .eml (Email Message)
- .msg (Outlook Message)
- .csv (Comma Separated Values)
- .html/.htm (HTML Documents)
- .md (Markdown)
- .json (JSON Data)
"""

import os
from abc import ABC, abstractmethod
from typing import List, Type
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredRTFLoader,
    UnstructuredHTMLLoader,
    UnstructuredEmailLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader,
)


class LoaderStrategy(ABC):
    """Abstract base class defining the interface for document loading strategies."""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        Load and parse documents from the given file path.

        Args:
            file_path: Path to the file to be loaded.

        Returns:
            List of Document objects extracted from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        pass


class PDFLoaderStrategy(LoaderStrategy):
    """Strategy for loading PDF documents."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects extracted from the PDF.
        """
        return UnstructuredPDFLoader(file_path).load()


class WordLoaderStrategy(LoaderStrategy):
    """Strategy for loading Microsoft Word documents."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a Word file (.docx or .doc).

        Args:
            file_path: Path to the Word file.

        Returns:
            List of Document objects extracted from the Word file.
        """
        return UnstructuredWordDocumentLoader(file_path).load()


class RTFLoaderStrategy(LoaderStrategy):
    """Strategy for loading RTF (Rich Text Format) documents."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from an RTF file.

        Args:
            file_path: Path to the RTF file.

        Returns:
            List of Document objects extracted from the RTF file.
        """
        return UnstructuredRTFLoader(file_path).load()


class HTMLLoaderStrategy(LoaderStrategy):
    """Strategy for loading HTML documents."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from an HTML file.

        Args:
            file_path: Path to the HTML file.

        Returns:
            List of Document objects extracted from the HTML file.
        """
        return UnstructuredHTMLLoader(file_path).load()


class EmailLoaderStrategy(LoaderStrategy):
    """Strategy for loading email documents (.eml or .msg)."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from an email file.

        Args:
            file_path: Path to the email file (.eml or .msg).

        Returns:
            List of Document objects extracted from the email.
        """
        return UnstructuredEmailLoader(file_path).load()


class MarkdownLoaderStrategy(LoaderStrategy):
    """Strategy for loading Markdown documents."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a Markdown file.

        Args:
            file_path: Path to the Markdown file (.md or .markdown).

        Returns:
            List of Document objects extracted from the Markdown file.
        """
        return UnstructuredMarkdownLoader(file_path).load()


class PowerPointLoaderStrategy(LoaderStrategy):
    """Strategy for loading PowerPoint presentations."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a PowerPoint file.

        Args:
            file_path: Path to the PowerPoint file (.pptx or .ppt).

        Returns:
            List of Document objects extracted from the PowerPoint file.
        """
        return UnstructuredPowerPointLoader(file_path).load()


class TextLoaderStrategy(LoaderStrategy):
    """Strategy for loading plain text documents."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a plain text file.

        Args:
            file_path: Path to the text file (.txt).

        Returns:
            List of Document objects extracted from the text file.
        """
        return TextLoader(file_path).load()


class CSVLoaderStrategy(LoaderStrategy):
    """Strategy for loading CSV files."""

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            List of Document objects extracted from the CSV file.
        """
        return CSVLoader(file_path).load()


class FallbackLoaderStrategy(LoaderStrategy):
    """Fallback strategy for loading documents of unknown types."""

    def load(self, file_path: str) -> List[Document]:
        """
        Attempt to load documents from a file of unknown type.

        Args:
            file_path: Path to the file.

        Returns:
            List of Document objects extracted from the file.
        """
        return UnstructuredFileLoader(file_path).load()


class UniversalDocumentLoader:
    """
    A universal document loader that automatically selects the appropriate loading strategy
    based on file extension.

    Attributes:
        file_path: Path to the file to be loaded.
        _strategy: The loading strategy selected for the file.
    """

    _strategy_map: dict[str, Type[LoaderStrategy]] = {
        ".pdf": PDFLoaderStrategy,
        ".docx": WordLoaderStrategy,
        ".doc": WordLoaderStrategy,
        ".rtf": RTFLoaderStrategy,
        ".html": HTMLLoaderStrategy,
        ".eml": EmailLoaderStrategy,
        ".msg": EmailLoaderStrategy,
        ".md": MarkdownLoaderStrategy,
        ".markdown": MarkdownLoaderStrategy,
        ".pptx": PowerPointLoaderStrategy,
        ".ppt": PowerPointLoaderStrategy,
        ".txt": TextLoaderStrategy,
        ".csv": CSVLoaderStrategy,
    }

    def __init__(self, file_path: str):
        """
        Initialize the UniversalDocumentLoader with the file to be loaded.

        Args:
            file_path: Path to the file to be loaded.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        self.file_path = file_path
        self._strategy = self._resolve_strategy()

    def _resolve_strategy(self) -> LoaderStrategy:
        """
        Determine the appropriate loading strategy based on file extension.

        Returns:
            An instance of the appropriate LoaderStrategy for the file type.
            Falls back to FallbackLoaderStrategy if extension is not recognized.
        """
        ext = os.path.splitext(self.file_path)[-1].lower()
        strategy_class = self._strategy_map.get(ext, FallbackLoaderStrategy)
        return strategy_class()

    def load(self) -> List[Document]:
        """
        Load documents using the selected strategy.

        Returns:
            List of Document objects extracted from the file.

        Raises:
            ValueError: If the file cannot be parsed by the selected strategy.
        """
        return self._strategy.load(self.file_path)
