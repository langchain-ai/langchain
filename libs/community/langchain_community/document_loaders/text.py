import logging
from pathlib import Path
from typing import Iterator, Optional, Union, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

logger = logging.getLogger(__name__)

COMMON_ENCODINGS = ["utf-8", "gb18030", "gbk", "gb2312", "iso-8859-1", "latin1", "cp936", "big5"]

class TextLoader(BaseLoader):
    """Load text file.


    Args:
        file_path: Path to the file to load.

        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.

        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
            
        fallback_encodings: List of encodings to try if the specified encoding fails.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
        fallback_encodings: Optional[List[str]] = None,
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
        self.fallback_encodings = fallback_encodings or COMMON_ENCODINGS

    def lazy_load(self) -> Iterator[Document]:
        """Load from file path."""
        text = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    logger.debug(f"Trying detected encoding: {encoding.encoding}")
                    try:
                        with open(self.file_path, encoding=encoding.encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
            
            if not text and self.fallback_encodings:
                for encoding in self.fallback_encodings:
                    if encoding == self.encoding:
                        continue 
                    logger.debug(f"Trying fallback encoding: {encoding}")
                    try:
                        with open(self.file_path, encoding=encoding) as f:
                            text = f.read()
                        logger.info(f"Successfully loaded file with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue

            if not text:
                raise RuntimeError(f"Error loading {self.file_path}: Unable to decode with any encoding") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        metadata = {"source": str(self.file_path)}
        yield Document(page_content=text, metadata=metadata)