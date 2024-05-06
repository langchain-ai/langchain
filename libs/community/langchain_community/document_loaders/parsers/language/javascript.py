from typing import Any, List, Tuple

from langchain_community.document_loaders.parsers.language.code_segmenter import (
    CodeSegmenter,
)


class JavaScriptSegmenter(CodeSegmenter):
    """Code segmenter for JavaScript."""

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines = self.code.splitlines()

        try:
            import esprima  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import esprima Python package. "
                "Please install it with `pip install esprima`."
            )

    def is_valid(self) -> bool:
        import esprima

        try:
            esprima.parseScript(self.code)
            return True
        except esprima.Error:
            return False

    def _extract_code(self, node: Any) -> str:
        start = node.loc.start.line - 1
        end = node.loc.end.line
        return "\n".join(self.source_lines[start:end])

    def extract_functions_classes(self) -> List[str]:
        import esprima

        tree = esprima.parseScript(self.code, loc=True)
        functions_classes = []

        for node in tree.body:
            if isinstance(
                node,
                (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration),
            ):
                functions_classes.append(self._extract_code(node))

        return functions_classes

    def simplify_code(self) -> str:
        import esprima

        tree = esprima.parseScript(self.code, loc=True)
        simplified_lines = self.source_lines[:]

        indices_to_del: List[Tuple[int, int]] = []
        for node in tree.body:
            if isinstance(
                node,
                (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration),
            ):
                start, end = node.loc.start.line - 1, node.loc.end.line
                simplified_lines[start] = f"// Code for: {simplified_lines[start]}"

                indices_to_del.append((start + 1, end))

        for start, end in reversed(indices_to_del):
            del simplified_lines[start + 0 : end]

        return "\n".join(line for line in simplified_lines)
