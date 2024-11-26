import zipfile
from pathlib import Path
from typing import Iterator, Union
from xml.etree.ElementTree import parse  # OK: user-must-opt-in

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
import logging

# Logger configuration
logger = logging.getLogger(__name__)

class HwpxLoader(BaseLoader):
    """
    Load `hwpx` files and convert their textual contents into LangChain Documents.

    This loader extracts only the text content from HWPX files. 
    Image files and non-textual content cannot be loaded.

    Args:
        file_path: Path to the HWPX file.

    Returns:
        Iterator of Document objects, each representing a section or piece of the HWPX file.
    """

    def __init__(self, file_path: Union[str, Path]):
        """Initialize with file path."""
        self.file_path = str(file_path)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load content from the HWPX file, yielding Documents."""
        try:
            with zipfile.ZipFile(self.file_path, "r") as hwpx_zip:
                file_list = hwpx_zip.namelist()
                content_files = [
                    x for x in file_list if x.startswith("Contents/sec") and x.endswith(".xml")
                ]

                for content_file in content_files:
                    try:
                        with hwpx_zip.open(content_file) as f:
                            tree = parse(f)
                            root = tree.getroot()

                            # Extract text from XML and yield as Document
                            text = self._extract_text_from_xml(root)
                            if text.strip():
                                yield Document(page_content=text, metadata={"source": content_file})

                    except Exception as e:
                        logger.error(f"Error processing file {content_file}: {e}")
        except zipfile.BadZipFile as e:
            logger.error(f"Error opening HWPX file {self.file_path}: Invalid zip format.") 
            raise RuntimeError(f"Error opening HWPX file {self.file_path}: Invalid zip format.") from e
        except Exception as e:
            logger.error(f"Unexpected error opening HWPX file {self.file_path}: {e}")
            raise RuntimeError(f"Error opening HWPX file {self.file_path}") from e

    def _extract_text_from_xml(self, root) -> str:
        """
        Extract meaningful text from the XML tree.

        Args:
            root: Root of the XML ElementTree.

        Returns:
            Combined text content of the XML.
        """
        text_segments = []
        for elem in root.iter():
            if elem.tag.endswith("t"):
                text_segments.append(elem.text or "")
        return "".join(text_segments)
