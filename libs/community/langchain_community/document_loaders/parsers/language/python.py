import ast
from typing import Any, List, Union

from langchain_community.document_loaders.parsers.language.code_segmenter import (
    CodeSegmenter,
)


class PythonSegmenter(CodeSegmenter):
    """Code segmenter for `Python`."""

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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # if the node has decorators, we want to include them in the code
            decorator_lines = [decorator.lineno for decorator in getattr(node, 'decorator_list', [])]
            start = min(decorator_lines + [node.lineno]) - 1 if decorator_lines else node.lineno - 1
        return "\n".join(self.source_lines[start:end])

    def _extract_function_classes_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return _extract_function_signature(node)
        elif isinstance(node, ast.ClassDef):
            return _extract_class_signature(node)
        else:
            raise NotImplementedError(f"Unsupported node type: {type(node)}")

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
                decorator_lines = [decorator.lineno for decorator in getattr(node, 'decorator_list', [])]
                start = min(decorator_lines + [node.lineno]) - 1 if decorator_lines else node.lineno - 1
                signature = self._extract_function_classes_signature(node)
                simplified_lines[start] = f"# Code for: {signature}"

                assert isinstance(node.end_lineno, int)
                for line_num in range(start + 1, node.end_lineno):
                    simplified_lines[line_num] = None  # type: ignore

        return "\n".join(line for line in simplified_lines if line is not None)


def _extract_function_signature(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    """Extract function signature from an `ast.FunctionDef` or `ast.AsyncFunctionDef` node.

    This is a patch for handling multiline function signatures.
    
    Args:
        node: The ast.FunctionDef or ast.AsyncFunctionDef node.

    Returns:
        The function signature, e.g. `def foo(a: int, b: str) -> None:`
    """
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return_signature = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){return_signature}:"


def _extract_class_signature(node: ast.ClassDef) -> str:
    """
    Extract class signature from an ast.ClassDef node.

    Args:
        node: The ast.ClassDef node.

    Returns:
        The class signature, e.g., `class MyClass(BaseClass1, BaseClass2):`
    """
    base_classes = ", ".join(ast.unparse(base) for base in node.bases)
    class_signature = f"class {node.name}({base_classes}):" if base_classes else f"class {node.name}:"
    return class_signature

