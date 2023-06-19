from typing import List
import ast

from langchain.document_loaders.language.language_parser import LanguageParser


class PythonParser(LanguageParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_lines = self.code.splitlines()

    def is_valid(self) -> bool:
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, node) -> str:
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

                for line_num in range(start + 1, node.end_lineno):
                    simplified_lines[line_num] = None

        return "\n".join(line for line in simplified_lines if line is not None)
