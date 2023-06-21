from typing import List, Dict, Any
from pathlib import Path

import tokenize

from langchain.docstore.document import Document
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.language.python import PythonParser
from langchain.document_loaders.language.javascript import JavaScriptParser


LANGUAGE_PARSERS: Dict[str, Dict[str, Any]] = {
    "py": {
        "parser": PythonParser,
        "language": "python",
    },
    "js": {
        "parser": JavaScriptParser,
        "language": "javascript",
    },
}


class LanguageLoader(TextLoader):
    """
    Load code files, using the respective parser.
    """

    def __init__(self, file_path: str, parser_threshold: int = 0):
        self.parser_threshold = parser_threshold
        self.file_extension = Path(file_path).suffix.lstrip(".")
        with open(file_path, "rb") as f:
            encoding, _ = tokenize.detect_encoding(f.readline)
        super().__init__(file_path=file_path, encoding=encoding)

    def load(self) -> List[Document]:
        [document] = super().load()

        if (
            self.parser_threshold >= len(document.page_content.splitlines())
            or self.file_extension not in LANGUAGE_PARSERS
        ):
            return [document]

        Parser = LANGUAGE_PARSERS[self.file_extension]["parser"]
        language = LANGUAGE_PARSERS[self.file_extension]["language"]

        parser = Parser(document.page_content)
        if not parser.is_valid():
            return [document]
        documents = []
        for functions_classes in parser.extract_functions_classes():
            documents.append(
                Document(
                    page_content=functions_classes,
                    metadata={
                        "source": document.metadata["source"],
                        "content_type": "functions_classes",
                        "language": language,
                    },
                )
            )
        documents.append(
            Document(
                page_content=parser.simplify_code(),
                metadata={
                    "source": document.metadata["source"],
                    "content_type": "simplified_code",
                    "language": language,
                },
            )
        )
        return documents
