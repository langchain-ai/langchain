import logging
from typing import List, Optional, cast

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class TextLoader(BaseLoader):
    """Load text files.


    Args:
        file_path: Path to the file to load.

        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.

        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
    """

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        text = ""
        with open(self.file_path, encoding=self.encoding) as f:
            try:
                text = f.read()
            except UnicodeDecodeError as e:
                if self.autodetect_encoding:
                    detected_encodings = detect_file_encodings(self.file_path)
                    for encoding in detected_encodings:
                        logger.debug("Trying encoding: ", encoding["encoding"])
                        try:
                            with open(
                                self.file_path, encoding=encoding["encoding"]
                            ) as f:
                                text = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                else:
                    raise RuntimeError(f"Error loading {self.file_path}") from e
            except Exception as e:
                raise RuntimeError(f"Error loading {self.file_path}") from e

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

def detect_file_encodings(file_path: str) -> List[dict]:
    """Try to detect the file encoding."""
    import chardet


    with open(file_path, "rb") as f:
        rawdata = f.read()
    encodings = chardet.detect_all(rawdata)


    if all(encoding["encoding"] is None for encoding in encodings):
        raise RuntimeError(f"Could not detect encoding for {file_path}")
    res = [encoding for encoding in encodings if encoding["encoding"] is not None]
    return cast(List[dict], res)
