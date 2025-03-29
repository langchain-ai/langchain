from abc import abstractmethod
from typing import TYPE_CHECKING, List

from langchain_community.document_loaders.parsers.language.code_segmenter import (
    CodeSegmenter,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Parser


class TreeSitterSegmenter(CodeSegmenter):
    """Abstract class for `CodeSegmenter`s that use the tree-sitter library."""

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines = self.code.splitlines()

        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import tree_sitter/tree_sitter_language_pack "
                "Python packages. Please install them with "
                "`pip install tree-sitter tree-sitter-language-pack`."
            )

    def is_valid(self) -> bool:
        language = self.get_language()
        error_query = language.query("(ERROR) @error")

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))

        return len(error_query.captures(tree.root_node)) == 0

    def _get_top_level_nodes(self) -> List["Node"]:
        language = self.get_language()
        query = language.query(self.get_chunk_query())
        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))
        captures = query.captures(tree.root_node)
        top_level_nodes = {}
        for node_type, nodes in captures.items():
            for node in nodes:
                cursor = node.parent
                is_child = False
                while cursor is not None:
                    if cursor.id in top_level_nodes:
                        is_child = True
                        break
                    cursor = cursor.parent
                if is_child:
                    continue
                top_level_nodes[node.id] = node

                children = node.children
                for child in children:
                    if child.id in top_level_nodes:
                        del top_level_nodes[child.id]
                    children.extend(child.children)
        top_level_nodes_list = list(top_level_nodes.values())
        top_level_nodes_list.sort(key=lambda n: n.start_point[0])
        return top_level_nodes_list

    def extract_functions_classes(self) -> List[str]:
        top_level_nodes = self._get_top_level_nodes()
        return [
            node.text.decode("UTF-8")
            for node in top_level_nodes
            if node.text is not None
        ]

    def simplify_code(self) -> str:
        simplified_lines = self.source_lines[:]
        top_level_nodes = self._get_top_level_nodes()
        for node in top_level_nodes:
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            simplified_lines[start_line] = self.make_line_comment(
                f"Code for: {self.source_lines[start_line]}"
            )

            for line_num in range(start_line + 1, end_line + 1):
                simplified_lines[line_num] = None  # type: ignore

        return "\n".join(line for line in simplified_lines if line is not None)

    @abstractmethod
    def get_parser(self) -> "Parser":
        raise NotImplementedError()

    @abstractmethod
    def get_language(self) -> "Language":
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_chunk_query(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def make_line_comment(self, text: str) -> str:
        raise NotImplementedError()  # pragma: no cover
