from pathlib import Path
from typing import Iterator, Union
from urllib.parse import urlparse

from langchain_core.documents import Document

from langchain_community.document_loaders.pdf import BaseLoader

DEFAULT_API = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"


class LLMSherpaFileLoader(BaseLoader):
    """Load Documents using `LLMSherpa`.

    LLMSherpaFileLoader use LayoutPDFReader, which is part of the LLMSherpa library.
    This tool is designed to parse PDFs while preserving their layout information,
    which is often lost when using most PDF to text parsers.

    Examples
    --------
    from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

    loader = LLMSherpaFileLoader(
        "example.pdf",
        strategy="chunks",
        llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
    )
    docs = loader.load()
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        new_indent_parser: bool = True,
        apply_ocr: bool = True,
        strategy: str = "chunks",
        llmsherpa_api_url: str = DEFAULT_API,
    ):
        """Initialize with a file path."""
        try:
            import llmsherpa  # noqa:F401
        except ImportError:
            raise ImportError(
                "llmsherpa package not found, please install it with "
                "`pip install llmsherpa`"
            )
        _valid_strategies = ["sections", "chunks", "html", "text"]
        if strategy not in _valid_strategies:
            raise ValueError(
                f"Got {strategy} for `strategy`, "
                f"but should be one of `{_valid_strategies}`"
            )
        # validate llmsherpa url
        if not self._is_valid_url(llmsherpa_api_url):
            raise ValueError(f"Invalid URL: {llmsherpa_api_url}")
        self.url = self._validate_llmsherpa_url(
            url=llmsherpa_api_url,
            new_indent_parser=new_indent_parser,
            apply_ocr=apply_ocr,
        )

        self.strategy = strategy
        self.file_path = str(file_path)

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def _validate_llmsherpa_url(
        url: str, new_indent_parser: bool = True, apply_ocr: bool = True
    ) -> str:
        """Check if the llmsherpa url is valid."""
        parsed = urlparse(url)
        valid_url = url
        if ("/api/parseDocument" not in parsed.path) and (
            "/api/document/developer/parseDocument" not in parsed.path
        ):
            raise ValueError(f"Invalid LLMSherpa URL: {url}")

        if "renderFormat=all" not in parsed.query:
            valid_url = valid_url + "?renderFormat=all"
        if new_indent_parser and "useNewIndentParser=true" not in parsed.query:
            valid_url = valid_url + "&useNewIndentParser=true"
        if apply_ocr and "applyOcr=yes" not in parsed.query:
            valid_url = valid_url + "&applyOcr=yes"

        return valid_url

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Load file."""
        from llmsherpa.readers import LayoutPDFReader

        docs_reader = LayoutPDFReader(self.url)
        doc = docs_reader.read_pdf(self.file_path)

        if self.strategy == "sections":
            yield from [
                Document(
                    page_content=section.to_text(include_children=True, recurse=True),
                    metadata={
                        "source": self.file_path,
                        "section_number": section_num,
                        "section_title": section.title,
                    },
                )
                for section_num, section in enumerate(doc.sections())
            ]
        if self.strategy == "chunks":
            yield from [
                Document(
                    page_content=chunk.to_context_text(),
                    metadata={
                        "source": self.file_path,
                        "chunk_number": chunk_num,
                        "chunk_type": chunk.tag,
                    },
                )
                for chunk_num, chunk in enumerate(doc.chunks())
            ]
        if self.strategy == "html":
            yield from [
                Document(
                    page_content=doc.to_html(),
                    metadata={
                        "source": self.file_path,
                    },
                )
            ]
        if self.strategy == "text":
            yield from [
                Document(
                    page_content=doc.to_text(),
                    metadata={
                        "source": self.file_path,
                    },
                )
            ]
