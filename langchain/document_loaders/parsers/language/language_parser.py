from typing import Any, Dict, Iterator, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.language.javascript import JavaScriptSegmenter
from langchain.document_loaders.parsers.language.python import PythonSegmenter
from langchain.text_splitter import Language

LANGUAGE_EXTENSIONS: Dict[str, str] = {
    "py": Language.PYTHON,
    "js": Language.JS,
}

LANGUAGE_SEGMENTERS: Dict[str, Any] = {
    Language.PYTHON: PythonSegmenter,
    Language.JS: JavaScriptSegmenter,
}


class LanguageParser(BaseBlobParser):
    """
    Parse code files, using the respective parser.
    """

    def __init__(self, language: Optional[Language] = None, parser_threshold: int = 0):
        self.language = language
        self.parser_threshold = parser_threshold

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        code = blob.as_string()

        language = self.language or (
            LANGUAGE_EXTENSIONS.get(blob.source.rsplit(".", 1)[-1])
            if isinstance(blob.source, str)
            else None
        )

        if language is None:
            yield Document(
                page_content=code,
                metadata={
                    "source": blob.source,
                },
            )
            return

        if self.parser_threshold >= len(code.splitlines()):
            yield Document(
                page_content=code,
                metadata={
                    "source": blob.source,
                    "language": language,
                },
            )
            return

        self.Segmenter = LANGUAGE_SEGMENTERS[language]
        segmenter = self.Segmenter(blob.as_string())
        if not segmenter.is_valid():
            yield Document(
                page_content=code,
                metadata={
                    "source": blob.source,
                },
            )
            return

        for functions_classes in segmenter.extract_functions_classes():
            yield Document(
                page_content=functions_classes,
                metadata={
                    "source": blob.source,
                    "content_type": "functions_classes",
                    "language": language,
                },
            )
        yield Document(
            page_content=segmenter.simplify_code(),
            metadata={
                "source": blob.source,
                "content_type": "simplified_code",
                "language": language,
            },
        )
