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

    # TODO: Actually implement these abstract methods

    def extract_functions_classes(self) -> List[str]:
        language = self.get_language()
        query = language.query(self.get_chunk_query())
        
        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding='UTF-8'))

        captures = query.captures(tree.root_node)
        return [node.text for (node, name) in captures]

    def simplify_code(self) -> str:
        # TODO
        pass
    #     import esprima

    #     tree = esprima.parseScript(self.code, loc=True)
    #     simplified_lines = self.source_lines[:]

    #     for node in tree.body:
    #         if isinstance(
    #             node,
    #             (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration),
    #         ):
    #             start = node.loc.start.line - 1
    #             simplified_lines[start] = f"// Code for: {simplified_lines[start]}"

    #             for line_num in range(start + 1, node.loc.end.line):
    #                 simplified_lines[line_num] = None  # type: ignore

    #     return "\n".join(line for line in simplified_lines if line is not None)

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
