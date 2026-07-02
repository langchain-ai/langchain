"""Intra-procedural AST taint analysis for tool implementations."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from langchain_core.security.models import SecurityFinding, Severity, SinkChannel

# Common hardcoded secret patterns.
HARDCODED_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"sk-proj-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    re.compile(r"xox[baprs]-[a-zA-Z0-9-]{10,}"),
)

_ENVIRONMENT_ATTRS = frozenset({"environ"})
_GETENV_NAMES = frozenset({"getenv", "get"})
_CREDENTIAL_ENV_HINTS = frozenset({
    "api_key",
    "api-key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "auth",
    "bearer",
    "access_key",
    "private_key",
})
_TAINT_SANITIZERS = frozenset({"bool", "len", "hash", "repr", "id"})
_LOGGING_METHODS = frozenset({
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "critical",
    "exception",
})
_REQUESTS_METHODS = frozenset({
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "request",
    "head",
    "options",
})
_REQUESTS_MODULES = frozenset({"requests", "httpx", "aiohttp"})


@dataclass
class _SinkHit:
    """Internal representation of a tainted value reaching a sink."""

    line: int
    column: int | None
    channel: SinkChannel
    rule_id: str
    severity: Severity
    message: str
    attack_flow: tuple[str, ...] = ()


@dataclass
class _TaintState:
    """Tracks tainted names within a function scope."""

    tainted: set[str] = field(default_factory=set)
    sinks: list[_SinkHit] = field(default_factory=list)


class ToolTaintAnalyzer(ast.NodeVisitor):
    """Analyze a single function body for credential leakage to agent-visible sinks."""

    def __init__(
        self,
        *,
        filename: str,
        tool_name: str | None,
        cross_modal: bool,
    ) -> None:
        self._filename = filename
        self._tool_name = tool_name
        self._cross_modal = cross_modal
        self._state = _TaintState()
        self._scope_stack: list[_TaintState] = []

    @property
    def sinks(self) -> list[_SinkHit]:
        """Collected sink hits from analysis."""
        return self._state.sinks

    def analyze(self, tree: ast.AST) -> list[SecurityFinding]:
        """Run analysis and return security findings."""
        self.visit(tree)
        return [self._to_finding(hit) for hit in self._state.sinks]

    def _to_finding(self, hit: _SinkHit) -> SecurityFinding:
        return SecurityFinding(
            rule_id=hit.rule_id,
            severity=hit.severity,
            message=hit.message,
            file=self._filename,
            line=hit.line,
            column=hit.column,
            channel=hit.channel,
            tool_name=self._tool_name,
            attack_flow=hit.attack_flow,
            cross_modal=self._cross_modal and hit.rule_id == "LC-TOOL-005",
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        value_tainted = self._expr_is_tainted(node.value)
        credential_source = self._is_credential_source(node.value)
        for target in node.targets:
            for name in self._extract_names(target):
                if credential_source or value_tainted:
                    self._state.tainted.add(name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            value_tainted = self._expr_is_tainted(node.value)
            credential_source = self._is_credential_source(node.value)
            for name in self._extract_names(node.target):
                if credential_source or value_tainted:
                    self._state.tainted.add(name)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and self._expr_is_tainted(node.value):
            self._record_sink(
                node,
                channel="return",
                rule_id=self._pick_rule("LC-TOOL-002", "LC-TOOL-005"),
                severity=Severity.HIGH,
                message="Credential may be returned to the agent observation loop.",
                flow_step="return tainted value → ToolMessage → LLM context",
            )
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self._check_call_sink(node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_call_sink(node)
        self.generic_visit(node)

    def _check_call_sink(self, node: ast.Call) -> None:
        func = node.func

        if self._is_print_call(func) and self._args_contain_taint(node):
            self._record_sink(
                node,
                channel="stdout",
                rule_id=self._pick_rule("LC-TOOL-001", "LC-TOOL-005"),
                severity=Severity.HIGH,
                message="Credential may be printed to stdout and captured by the agent.",
                flow_step="print tainted value → stdout → agent context",
            )
            return

        if self._is_logging_call(func) and self._args_contain_taint(node):
            self._record_sink(
                node,
                channel="logging",
                rule_id=self._pick_rule("LC-TOOL-003", "LC-TOOL-005"),
                severity=Severity.MEDIUM,
                message="Credential may be logged and exposed to the agent.",
                flow_step="log tainted value → logs → agent context",
            )
            return

        if self._is_network_call(func) and self._network_call_tainted(node):
            self._record_sink(
                node,
                channel="network",
                rule_id=self._pick_rule("LC-TOOL-004", "LC-TOOL-005"),
                severity=Severity.CRITICAL,
                message="Credential may be sent over the network from tool code.",
                flow_step="network call with tainted credential",
            )

    def _pick_rule(self, base_rule: str, cross_modal_rule: str) -> str:
        return cross_modal_rule if self._cross_modal else base_rule

    def _record_sink(
        self,
        node: ast.AST,
        *,
        channel: SinkChannel,
        rule_id: str,
        severity: Severity,
        message: str,
        flow_step: str,
    ) -> None:
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", None)
        attack_flow: tuple[str, ...] = ()
        if self._cross_modal:
            attack_flow = (
                "tool description references credentials",
                flow_step,
            )
        self._state.sinks.append(
            _SinkHit(
                line=line,
                column=col,
                channel=channel,
                rule_id=rule_id,
                severity=severity,
                message=message,
                attack_flow=attack_flow,
            )
        )

    def _push_scope(self) -> None:
        self._scope_stack.append(self._state)
        self._state = _TaintState()

    def _pop_scope(self) -> None:
        finished = self._state
        self._state = self._scope_stack.pop()
        self._state.sinks.extend(finished.sinks)

    def _extract_names(self, node: ast.AST) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Tuple):
            names: list[str] = []
            for elt in node.elts:
                names.extend(self._extract_names(elt))
            return names
        return []

    def _expr_is_tainted(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name):
            return node.id in self._state.tainted
        if isinstance(node, ast.JoinedStr):
            return any(self._expr_is_tainted(value) for value in node.values)
        if isinstance(node, ast.FormattedValue):
            return self._expr_is_tainted(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._expr_is_tainted(node.left) or self._expr_is_tainted(node.right)
        if isinstance(node, ast.Dict):
            return any(
                self._expr_is_tainted(value)
                for value in node.values
                if value is not None
            )
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            return any(self._expr_is_tainted(elt) for elt in node.elts)
        if isinstance(node, ast.Call):
            if self._is_sanitizer_call(node):
                return False
            return self._is_credential_source(node) or any(
                self._expr_is_tainted(arg) for arg in node.args
            ) or any(
                self._expr_is_tainted(kw.value)
                for kw in node.keywords
                if kw.value is not None
            )
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id in self._state.tainted
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id in self._state.tainted
        return False

    def _args_contain_taint(self, node: ast.Call) -> bool:
        return any(self._expr_is_tainted(arg) for arg in node.args) or any(
            self._expr_is_tainted(kw.value) for kw in node.keywords if kw.value is not None
        )

    def _network_call_tainted(self, node: ast.Call) -> bool:
        tainted_kwargs = {"data", "json", "headers", "params", "body", "auth", "content"}
        for kw in node.keywords:
            if kw.arg in tainted_kwargs and self._expr_is_tainted(kw.value):
                return True
        return self._args_contain_taint(node)

    def _is_sanitizer_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _TAINT_SANITIZERS:
            return bool(node.args)
        if isinstance(func, ast.Attribute) and func.attr in _TAINT_SANITIZERS:
            return bool(node.args)
        return False

    def _is_credential_source(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False

        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = func.value.id
            attr = func.attr
            if module == "os" and attr == "getenv":
                return self._env_arg_looks_sensitive(node)
            if module == "os" and attr in _GETENV_NAMES and self._looks_like_environ(func):
                return self._env_arg_looks_sensitive(node)
        if isinstance(func, ast.Attribute) and func.attr in _ENVIRONMENT_ATTRS:
            return True
        if isinstance(func, ast.Name) and func.id == "getenv":
            return self._env_arg_looks_sensitive(node)
        return False

    def _looks_like_environ(self, func: ast.Attribute) -> bool:
        return (
            isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
            and func.value.attr == "environ"
        )

    def _env_arg_looks_sensitive(self, node: ast.Call) -> bool:
        if not node.args:
            return True
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            normalized = first.value.lower().replace("-", "_")
            return any(hint in normalized for hint in _CREDENTIAL_ENV_HINTS)
        return True

    def _is_print_call(self, func: ast.AST) -> bool:
        return (isinstance(func, ast.Name) and func.id == "print") or (
            isinstance(func, ast.Attribute)
            and func.attr == "write"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "stdout"
        )

    def _is_logging_call(self, func: ast.AST) -> bool:
        if isinstance(func, ast.Attribute) and func.attr in _LOGGING_METHODS:
            if isinstance(func.value, ast.Name) and func.value.id == "logging":
                return True
            if isinstance(func.value, ast.Attribute) and func.value.attr == "logger":
                return True
            if isinstance(func.value, ast.Name) and func.value.id == "logger":
                return True
        return False

    def _is_network_call(self, func: ast.AST) -> bool:
        if isinstance(func, ast.Attribute) and func.attr in _REQUESTS_METHODS:
            if isinstance(func.value, ast.Name) and func.value.id in _REQUESTS_MODULES:
                return True
            if (
                isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in _REQUESTS_MODULES
            ):
                return True
        return False


def analyze_function_source(
    source: str,
    *,
    filename: str,
    tool_name: str | None,
    cross_modal: bool,
) -> list[SecurityFinding]:
    """Analyze a Python function source string for credential leakage."""
    tree = ast.parse(source, filename=filename)
    analyzer = ToolTaintAnalyzer(
        filename=filename,
        tool_name=tool_name,
        cross_modal=cross_modal,
    )
    findings = analyzer.analyze(tree)
    findings.extend(
        _find_hardcoded_secrets(tree, filename=filename, tool_name=tool_name)
    )
    return findings


def _find_hardcoded_secrets(
    tree: ast.AST,
    *,
    filename: str,
    tool_name: str | None,
) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        for pattern in HARDCODED_SECRET_PATTERNS:
            if pattern.search(node.value):
                findings.append(
                    SecurityFinding(
                        rule_id="LC-TOOL-006",
                        severity=Severity.CRITICAL,
                        message="Hardcoded secret detected in tool implementation.",
                        file=filename,
                        line=node.lineno,
                        column=node.col_offset,
                        channel="hardcoded_secret",
                        tool_name=tool_name,
                    )
                )
                break
    return findings


def find_tool_functions(source: str, *, filename: str) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Find top-level functions decorated with ``@tool`` in a module."""
    tree = ast.parse(source, filename=filename)
    decorated: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _has_tool_decorator(node):
            decorated.append((node.name, node))
    return decorated


def _has_tool_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "tool":
            return True
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            if decorator.func.id == "tool":
                return True
    return False


def function_source_from_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    module_source: str,
) -> str:
    """Extract source for a function AST node from module source."""
    lines = module_source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "".join(lines[start:end])
