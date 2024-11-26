from pathlib import Path
from typing import Iterator, Union

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

import zipfile

from xml.etree.ElementTree import parse # OK: user-must-opt-in


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
        with zipfile.ZipFile(self.file_path, "r") as hwpx_zip:
            file_list = hwpx_zip.namelist()
            content_files = [x for x in file_list if x.startswith("Contents/sec") and x.endswith(".xml")]

            for content_file in content_files:
                with hwpx_zip.open(content_file) as f:
                    tree = parse(f)
                    root = tree.getroot()

                    # Extract text from XML and yield as Document
                    text = "".join([elem.text or "" for elem in root.iter() if elem.tag.endswith("t")]).strip()
                    if text:
                        yield Document(page_content=text, metadata={"source": content_file})
