from typing import Any, List

from langchain.document_loaders.parsers.language.code_segmenter import CodeSegmenter


class CPPSegmenter(CodeSegmenter):
    """Code segmenter for C++."""

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
        from tree_sitter import Language, Parser
        language = Language('/tmp/tree-sitter-cpp.so', 'cpp')
        parser = Parser()
        parser.set_language(language)
        try:
            parser.parse(bytes(self.code, encoding='UTF-8'))
            return True
            # TODO: Find real error type and only catch that
        except:
            raise

    # TODO: Actually implement these abstract methods

    def extract_functions_classes(self) -> List[str]:
        pass
    #     import esprima

    #     tree = esprima.parseScript(self.code, loc=True)
    #     functions_classes = []

    #     for node in tree.body:
    #         if isinstance(
    #             node,
    #             (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration),
    #         ):
    #             functions_classes.append(self._extract_code(node))

    #     return functions_classes

    def simplify_code(self) -> str:
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
