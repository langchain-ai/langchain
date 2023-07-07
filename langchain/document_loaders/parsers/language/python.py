import ast
from typing import Any, List

from langchain.document_loaders.parsers.language.code_segmenter import CodeSegmenter


class PythonSegmenter(CodeSegmenter):
    """The code segmenter for Python."""

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines = self.code.splitlines()

    def is_valid(self) -> bool:
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, node: Any) -> str:
        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(self.source_lines[start:end])

    def extract_functions_classes(self) -> List[str]:
        tree = ast.parse(self.code)
        functions_classes = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                functions_classes.append(self._extract_code(node))

        return functions_classes

    def simplify_code(self) -> str:
        tree = ast.parse(self.code)
        simplified_lines = self.source_lines[:]

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                simplified_lines[start] = f"# Code for: {simplified_lines[start]}"

                assert isinstance(node.end_lineno, int)
                for line_num in range(start + 1, node.end_lineno):
                    simplified_lines[line_num] = None  # type: ignore

        return "\n".join(line for line in simplified_lines if line is not None)
