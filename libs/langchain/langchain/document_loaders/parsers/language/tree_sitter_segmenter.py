from abc import abstractmethod
from typing import List

from langchain.document_loaders.parsers.language.code_segmenter import CodeSegmenter


class TreeSitterSegmenter(CodeSegmenter):
    """Abstract class for `CodeSegmenter`s that use the tree-sitter library."""

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines = self.code.splitlines()

        try:
            import tree_sitter  # noqa: F401
            
            # TODO: Document this as a manual step if needed
            # https://til.simonwillison.net/python/tree-sitter
            
            # The below auto-downloading business could potentially work when using the library,
            # but it's blocked while running tests

            # from urllib.request import urlretrieve

            # # TODO: Make the path configurable
            # # TODO: Error handling
            # urlretrieve("https://github.com/tree-sitter/tree-sitter-cpp/archive/refs/heads/master.zip", "/tmp/tree-sitter-cpp")
            
            # tree_sitter.Language.build_library('/tmp/tree-sitter-cpp.so', ['/tmp/tree-sitter-cpp'])
        except ImportError as e:
            # TODO: Real error message
            raise ImportError(
                str(e)
            )

    def is_valid(self) -> bool:
        parser = self.get_parser()
        try:
            parser.parse(bytes(self.code, encoding='UTF-8'))
            return True
            # TODO: Find real error type and only catch that
        except:
            raise

    def extract_functions_classes(self) -> List[str]:
        language = self.get_language()
        query = language.query(self.get_chunk_query())
        
        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding='UTF-8'))

        captures = query.captures(tree.root_node)

        return [node.text.decode('UTF-8') for (node, name) in captures]

    def simplify_code(self) -> str:
        language = self.get_language()
        query = language.query(self.get_chunk_query())

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding='UTF-8'))

        # TODO: Track which lines already marked & blanked,
        #       to keep from processing chunks inside other chunks
        simplified_lines = self.source_lines[:]
        for (node, name) in query.captures(tree.root_node):
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            simplified_lines[start_line] = self.make_line_comment(f"Code for: {self.source_lines[start_line]}")

            for line_num in range(start_line + 1, end_line + 1):
                simplified_lines[line_num] = None  # type: ignore

        return "\n".join(line for line in simplified_lines if line is not None)

    def get_language(self):
        from tree_sitter import Language
        return Language('/tmp/tree-sitter-cpp.so', 'cpp')

    def get_parser(self):
        from tree_sitter import Parser
        parser = Parser()
        parser.set_language(self.get_language())
        return parser

    @abstractmethod
    def get_chunk_query(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def make_line_comment(self, text: str) -> str:
        raise NotImplementedError()  # pragma: no cover
