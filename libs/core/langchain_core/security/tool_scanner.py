"""Static security scanner for LangChain custom tools."""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core._api import beta
from langchain_core.security._nl_extract import (
    extract_nl_from_tool,
    nl_mentions_credentials,
)
from langchain_core.security._taint import (
    analyze_function_source,
    find_tool_functions,
    function_source_from_node,
)
from langchain_core.security.models import ScanResult, SecurityFinding, Severity

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.tools.base import BaseTool


@beta(message="Tool security scanning is in beta and may change.")
class ToolSecurityScanner:
    """Static analyzer for credential leakage in LangChain tools.

    This scanner performs intra-procedural analysis of tool implementations to
    detect when credentials may flow into agent-visible sinks such as return
    values, stdout, logs, or outbound network calls.

    When a tool's natural-language description references credentials and the
    implementation contains a matching leak path, a cross-modal finding is
    reported (``LC-TOOL-005``).
    """

    def scan_tools(self, tools: Sequence[BaseTool]) -> ScanResult:
        """Scan instantiated LangChain tools.

        Args:
            tools: Tool instances to analyze.

        Returns:
            Aggregated scan result.
        """
        result = ScanResult()
        for tool in tools:
            result.merge(self.scan_tool(tool))
        return result

    def scan_tool(self, tool: BaseTool) -> ScanResult:
        """Scan a single LangChain tool instance.

        Args:
            tool: Tool instance to analyze.

        Returns:
            Scan result for the tool.
        """
        func = _resolve_callable(tool)
        if func is None:
            return ScanResult()

        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            return ScanResult()

        source = textwrap.dedent(source)
        nl_text = extract_nl_from_tool(tool)
        cross_modal = nl_mentions_credentials(nl_text)
        filename = _callable_filename(func)
        findings = analyze_function_source(
            source,
            filename=filename,
            tool_name=getattr(tool, "name", None),
            cross_modal=cross_modal,
        )
        return ScanResult(findings=findings)

    def scan_callable(
        self,
        func: Callable[..., object],
        *,
        description: str = "",
        tool_name: str | None = None,
    ) -> ScanResult:
        """Scan a plain Python callable as if it were a tool implementation.

        Args:
            func: Function or coroutine to analyze.
            description: Optional natural-language description for cross-modal checks.
            tool_name: Optional tool name for reporting.

        Returns:
            Scan result for the callable.
        """
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            return ScanResult()

        source = textwrap.dedent(source)
        cross_modal = nl_mentions_credentials(description)
        findings = analyze_function_source(
            source,
            filename=_callable_filename(func),
            tool_name=tool_name or getattr(func, "__name__", None),
            cross_modal=cross_modal,
        )
        return ScanResult(findings=findings)

    def scan_path(self, path: str | Path) -> ScanResult:
        """Scan Python files for ``@tool``-decorated functions.

        Args:
            path: File or directory path to scan.

        Returns:
            Aggregated scan result.
        """
        target = Path(path)
        result = ScanResult()
        if target.is_file():
            if target.suffix == ".py":
                result.merge(self._scan_file(target))
            return result

        for file_path in sorted(target.rglob("*.py")):
            result.merge(self._scan_file(file_path))
        return result

    def scan_source(self, source: str, *, filename: str = "<string>") -> ScanResult:
        """Scan a Python source string for ``@tool``-decorated functions.

        Args:
            source: Python module source code.
            filename: Logical filename used in findings.

        Returns:
            Aggregated scan result.
        """
        result = ScanResult()
        for tool_name, node in find_tool_functions(source, filename=filename):
            fn_source = function_source_from_node(node, module_source=source)
            docstring = ast_get_docstring(node)
            cross_modal = nl_mentions_credentials(docstring or "")
            findings = analyze_function_source(
                fn_source,
                filename=filename,
                tool_name=tool_name,
                cross_modal=cross_modal,
            )
            result.findings.extend(findings)
        return result

    def _scan_file(self, file_path: Path) -> ScanResult:
        source = file_path.read_text(encoding="utf-8")
        result = self.scan_source(source, filename=str(file_path))
        return result


def _resolve_callable(tool: BaseTool) -> Callable[..., object] | None:
    func = getattr(tool, "func", None)
    if callable(func):
        return func
    coroutine = getattr(tool, "coroutine", None)
    if callable(coroutine):
        return coroutine
    return None


def _callable_filename(func: Callable[..., object]) -> str:
    module = inspect.getmodule(func)
    if module is not None and getattr(module, "__file__", None):
        return str(module.__file__)
    code = getattr(func, "__code__", None)
    if code is not None and code.co_filename:
        return code.co_filename
    return "<string>"


def ast_get_docstring(node: object) -> str | None:
    """Return the docstring for an AST function node."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ast.get_docstring(node)
    return None


__all__ = [
    "ScanResult",
    "SecurityFinding",
    "Severity",
    "ToolSecurityScanner",
]
