from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.language.c import CSegmenter
from langchain_community.document_loaders.parsers.language.cobol import CobolSegmenter
from langchain_community.document_loaders.parsers.language.cpp import CPPSegmenter
from langchain_community.document_loaders.parsers.language.csharp import CSharpSegmenter
from langchain_community.document_loaders.parsers.language.go import GoSegmenter
from langchain_community.document_loaders.parsers.language.java import JavaSegmenter
from langchain_community.document_loaders.parsers.language.javascript import (
    JavaScriptSegmenter,
)
from langchain_community.document_loaders.parsers.language.kotlin import KotlinSegmenter
from langchain_community.document_loaders.parsers.language.lua import LuaSegmenter
from langchain_community.document_loaders.parsers.language.perl import PerlSegmenter
from langchain_community.document_loaders.parsers.language.python import PythonSegmenter
from langchain_community.document_loaders.parsers.language.ruby import RubySegmenter
from langchain_community.document_loaders.parsers.language.rust import RustSegmenter
from langchain_community.document_loaders.parsers.language.scala import ScalaSegmenter
from langchain_community.document_loaders.parsers.language.typescript import (
    TypeScriptSegmenter,
)

if TYPE_CHECKING:
    from langchain.langchain.text_splitter import Language

try:
    from langchain.langchain.text_splitter import Language

    LANGUAGE_EXTENSIONS: Dict[str, str] = {
        "py": Language.PYTHON,
        "js": Language.JS,
        "cobol": Language.COBOL,
        "c": Language.C,
        "cpp": Language.CPP,
        "cs": Language.CSHARP,
        "rb": Language.RUBY,
        "scala": Language.SCALA,
        "rs": Language.RUST,
        "go": Language.GO,
        "kt": Language.KOTLIN,
        "lua": Language.LUA,
        "pl": Language.PERL,
        "ts": Language.TS,
        "java": Language.JAVA,
    }

    LANGUAGE_SEGMENTERS: Dict[str, Any] = {
        Language.PYTHON: PythonSegmenter,
        Language.JS: JavaScriptSegmenter,
        Language.COBOL: CobolSegmenter,
        Language.C: CSegmenter,
        Language.CPP: CPPSegmenter,
        Language.CSHARP: CSharpSegmenter,
        Language.RUBY: RubySegmenter,
        Language.RUST: RustSegmenter,
        Language.SCALA: ScalaSegmenter,
        Language.GO: GoSegmenter,
        Language.KOTLIN: KotlinSegmenter,
        Language.LUA: LuaSegmenter,
        Language.PERL: PerlSegmenter,
        Language.TS: TypeScriptSegmenter,
        Language.JAVA: JavaSegmenter,
    }
except ImportError:
    LANGUAGE_EXTENSIONS = {}
    LANGUAGE_SEGMENTERS = {}


class LanguageParser(BaseBlobParser):
    """Parse using the respective programming language syntax.

    Each top-level function and class in the code is loaded into separate documents.
    Furthermore, an extra document is generated, containing the remaining top-level code
    that excludes the already segmented functions and classes.

    This approach can potentially improve the accuracy of QA models over source code.

    The supported languages for code parsing are:

    - C (*)
    - C++ (*)
    - C# (*)
    - COBOL
    - Go (*)
    - Java (*)
    - JavaScript (requires package `esprima`)
    - Kotlin (*)
    - Lua (*)
    - Perl (*)
    - Python
    - Ruby (*)
    - Rust (*)
    - Scala (*)
    - TypeScript (*)

    Items marked with (*) require the packages `tree_sitter` and
    `tree_sitter_languages`. It is straightforward to add support for additional
    languages using `tree_sitter`, although this currently requires modifying LangChain.

    The language used for parsing can be configured, along with the minimum number of
    lines required to activate the splitting based on syntax.

    If a language is not explicitly specified, `LanguageParser` will infer one from
    filename extensions, if present.

    Examples:

       .. code-block:: python

            from langchain.text_splitter.Language
            from langchain_community.document_loaders.generic import GenericLoader
            from langchain_community.document_loaders.parsers import LanguageParser

            loader = GenericLoader.from_filesystem(
                "./code",
                glob="**/*",
                suffixes=[".py", ".js"],
                parser=LanguageParser()
            )
            docs = loader.load()

        Example instantiations to manually select the language:

        .. code-block:: python

            from langchain.text_splitter import Language

            loader = GenericLoader.from_filesystem(
                "./code",
                glob="**/*",
                suffixes=[".py"],
                parser=LanguageParser(language=Language.PYTHON)
            )

        Example instantiations to set number of lines threshold:

        .. code-block:: python

            loader = GenericLoader.from_filesystem(
                "./code",
                glob="**/*",
                suffixes=[".py"],
                parser=LanguageParser(parser_threshold=200)
            )
    """

    def __init__(self, language: Optional[Language] = None, parser_threshold: int = 0):
        """
        Language parser that split code using the respective language syntax.

        Args:
            language: If None (default), it will try to infer language from source.
            parser_threshold: Minimum lines needed to activate parsing (0 by default).
        """
        if language and language not in LANGUAGE_SEGMENTERS:
            raise Exception(f"No parser available for {language}")
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
