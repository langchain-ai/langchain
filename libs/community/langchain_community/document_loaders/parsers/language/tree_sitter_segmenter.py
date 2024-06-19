from abc import abstractmethod
from typing import TYPE_CHECKING, List

from langchain_community.document_loaders.parsers.language.code_segmenter import (
    CodeSegmenter,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Parser


class TreeSitterSegmenter(CodeSegmenter):
    """Abstract class for `CodeSegmenter`s that use the tree-sitter library."""

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines = self.code.splitlines()

        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_languages  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import tree_sitter/tree_sitter_languages Python packages. "
                "Please install them with "
                "`pip install tree-sitter tree-sitter-languages`."
            )

    def is_valid(self) -> bool:
        language = self.get_language()
        error_query = language.query("(ERROR) @error")

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))

        return len(error_query.captures(tree.root_node)) == 0

    def extract_functions_classes(self) -> List[str]:
        language = self.get_language()
        query = language.query(self.get_chunk_query())

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))
        captures = query.captures(tree.root_node)

        processed_lines = set()
        chunks = []

        for node, name in captures:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            lines = list(range(start_line, end_line + 1))

            if any(line in processed_lines for line in lines):
                continue

            processed_lines.update(lines)
            chunk_text = node.text.decode("UTF-8")
            chunks.append(chunk_text)

        return chunks

    def simplify_code(self) -> str:
        language = self.get_language()
        query = language.query(self.get_chunk_query())

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))
        processed_lines = set()

        simplified_lines = self.source_lines[:]
        for node, name in query.captures(tree.root_node):
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            lines = list(range(start_line, end_line + 1))
            if any(line in processed_lines for line in lines):
                continue

            simplified_lines[start_line] = self.make_line_comment(
                f"Code for: {self.source_lines[start_line]}"
            )

            for line_num in range(start_line + 1, end_line + 1):
                simplified_lines[line_num] = None  # type: ignore

            processed_lines.update(lines)

        return "\n".join(line for line in simplified_lines if line is not None)

    def get_parser(self) -> "Parser":
        from tree_sitter import Parser

        parser = Parser()
        parser.set_language(self.get_language())
        return parser

    @abstractmethod
    def get_language(self) -> "Language":
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_chunk_query(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def make_line_comment(self, text: str) -> str:
        raise NotImplementedError()  # pragma: no cover
