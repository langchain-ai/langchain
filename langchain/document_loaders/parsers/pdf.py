"""PDF parsers."""
from typing import Generator

from langchain.document_loaders.base import BaseBlobParser, Blob
from langchain.schema import Document


class PDFMinerParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Lazy parse PDF using PDFMiner."""
        try:
            from pdfminer.high_level import extract_text  # noqa:F401
        except ImportError:
            raise ValueError(
                "pdfminer package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        text = extract_text(blob.data)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
        raise NotImplementedError()
    
    