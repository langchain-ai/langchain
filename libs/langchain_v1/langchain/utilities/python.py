"""Python REPL utilities, including a security-hardened variant."""

from __future__ import annotations

import ast
import functools
import logging
import multiprocessing
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# SECURITY: Default-deny pattern per OWASP
DEFAULT_DENIED_IMPORTS: frozenset[str] = frozenset(
    {
        "__builtin__",
        "builtins",
        "glob",
        "httpx",
        "os",
        "paramiko",
        "requests",
        "shutil",
        "smtplib",
        "socket",
        "subprocess",
        "sys",
        "telnetlib",
        "tempfile",
        "urllib",
        "urllib3",
    }
)

DEFAULT_DENIED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
        "vars",
    }
)

SENSITIVE_PATHS: tuple[str, ...] = ("/etc", "/proc", "/sys", "/root", "/home")


@functools.lru_cache(maxsize=None)
def _warn_once() -> None:
    """Emit a one-time warning about the dangers of arbitrary code execution."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    model_config = ConfigDict(populate_by_name=True)

    globals: Optional[dict[str, Any]] = Field(default_factory=dict, alias="_globals")  # noqa: A003
    locals: Optional[dict[str, Any]] = Field(default_factory=dict, alias="_locals")  # noqa: A003

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the Python REPL.

        Strips leading/trailing whitespace, backticks, and an optional ``python``
        keyword that some models prepend to code blocks.

        Args:
            query: The raw query string to sanitize.

        Returns:
            The sanitized query string.
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    @classmethod
    def worker(
        cls,
        command: str,
        globals: Optional[dict[str, Any]],  # noqa: A002
        locals: Optional[dict[str, Any]],  # noqa: A002
        queue: multiprocessing.Queue,  # type: ignore[type-arg]
    ) -> None:
        """Execute *command* and place its stdout output onto *queue*."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            cleaned_command = cls.sanitize_input(command)
            exec(cleaned_command, globals, locals)  # noqa: S102
            sys.stdout = old_stdout
            queue.put(mystdout.getvalue())
        except Exception as exc:
            sys.stdout = old_stdout
            queue.put(repr(exc))

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """Run *command* and return anything printed to stdout.

        Args:
            command: Python source code to execute.
            timeout: Maximum execution time in seconds. ``None`` means no limit.

        Returns:
            Captured stdout output, or ``"Execution timed out"`` if the timeout
            was exceeded.
        """
        _warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]

        if timeout is not None:
            p = multiprocessing.Process(
                target=self.worker,
                args=(command, self.globals, self.locals, queue),
            )
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                return "Execution timed out"
        else:
            self.worker(command, self.globals, self.locals, queue)

        return queue.get()


class SafePythonREPL(PythonREPL):
    """A security-hardened variant of `PythonREPL` that pre-validates code via AST analysis.

    This class adds a defence-in-depth layer by parsing incoming Python code and
    rejecting or flagging patterns commonly associated with privilege escalation,
    data exfiltration, or environment inspection. It is **not** a sandbox — isolated
    execution should still be arranged at the OS or container level. This validator
    is intended to catch accidental or adversarial misuse of dangerous built-ins
    before the code reaches the interpreter.

    # SECURITY: Default-deny pattern per OWASP

    Args:
        mode: Controls what happens when a policy violation is detected.
            ``"block"`` (the default) raises `ValueError` immediately and prevents
            execution. ``"warn"`` logs the violation and **allows execution to
            continue** — this should only be used in development or testing
            environments where visibility into LLM-generated code is needed without
            hard stops.

            !!! warning
                ``mode="block"`` is the only safe default. Use ``mode="warn"``
                **only** in development environments with proper logging in place.
                Blocked patterns indicate potential attack vectors that must be
                reviewed and rejected before reaching production.
                Never use ``mode="warn"`` in production.

        allowed_imports: Import names explicitly permitted even if they appear in
            the denied list. For example, ``allowed_imports={"numpy", "pandas"}``
            allows those modules to be imported without triggering a policy
            violation.
        denied_imports: Replaces (not extends) the built-in denied-imports set.
            To extend the defaults, pass
            ``denied_imports=DEFAULT_DENIED_IMPORTS | {"my_lib"}``.
        denied_functions: Replaces (not extends) the built-in denied-functions
            set. Same semantics as *denied_imports*.
        timeout: Default execution timeout in seconds applied to every `run()`
            call. Individual calls may override this. Set to ``None`` to disable.

    Examples:
        Block-mode — safe for all environments (default)::

            repl = SafePythonREPL()

            # Raises ValueError — "import os" violates the default policy.
            repl.run("import os; os.getcwd()")

            # Safe — pure arithmetic passes through.
            result = repl.run("print(2 + 2)")  # "4\\n"

            # Opt-in to numpy while keeping everything else blocked.
            repl = SafePythonREPL(allowed_imports={"numpy"})
            result = repl.run("import numpy as np; print(np.pi)")

        Warn-mode — development/testing only, **UNSAFE**::

            import logging
            logging.basicConfig(level=logging.WARNING)

            repl = SafePythonREPL(mode="warn")
            # Logs a WARNING then executes anyway.
            repl.run("import os; print(os.getcwd())")
    """

    mode: Literal["block", "warn"] = "block"
    allowed_imports: Optional[set[str]] = None
    denied_imports: Optional[set[str]] = None
    denied_functions: Optional[set[str]] = None
    timeout: Optional[int] = None

    def _get_effective_denied_imports(self) -> set[str]:
        base: set[str] = (
            self.denied_imports
            if self.denied_imports is not None
            else set(DEFAULT_DENIED_IMPORTS)
        )
        return base - (self.allowed_imports or set())

    def _get_effective_denied_functions(self) -> set[str]:
        return (
            self.denied_functions
            if self.denied_functions is not None
            else set(DEFAULT_DENIED_FUNCTIONS)
        )

    def _validate_code_safety(self, code: str) -> dict[str, Any]:
        """Parse and walk the code's AST, returning a safety assessment.

        Args:
            code: Raw Python source code to analyse.

        Returns:
            A dict with keys ``"safe"`` (bool) and ``"reason"`` (str | None).
            ``"reason"`` is ``None`` when the code is deemed safe.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Unparseable code is not necessarily unsafe; let exec() surface the error.
            return {"safe": True, "reason": None}

        effective_denied_imports = self._get_effective_denied_imports()
        effective_denied_functions = self._get_effective_denied_functions()

        for node in ast.walk(tree):
            # --- Import statements ---
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    module_names = [alias.name for alias in node.names]
                else:
                    module_names = [node.module] if node.module else []

                for name in module_names:
                    root = name.split(".")[0] if name else ""
                    if root in effective_denied_imports:
                        return {
                            "safe": False,
                            "reason": f"import of denied module '{root}'",
                        }

            # --- Function calls ---
            if isinstance(node, ast.Call):
                # Direct built-in calls: eval(...), exec(...), open(...), etc.
                if isinstance(node.func, ast.Name):
                    if node.func.id in effective_denied_functions:
                        return {
                            "safe": False,
                            "reason": f"call to denied built-in '{node.func.id}'",
                        }

                # Attribute calls on denied modules: socket.connect(), urllib.request.urlopen()
                if isinstance(node.func, ast.Attribute):
                    obj = node.func.value
                    if isinstance(obj, ast.Name) and obj.id in effective_denied_imports:
                        return {
                            "safe": False,
                            "reason": f"attribute access on denied module '{obj.id}'",
                        }

                # Sensitive-path guard for open() — defence-in-depth even when
                # 'open' is removed from denied_functions via customisation.
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(
                            first_arg.value, str
                        ):
                            for sensitive in SENSITIVE_PATHS:
                                if first_arg.value.startswith(sensitive):
                                    return {
                                        "safe": False,
                                        "reason": (
                                            f"open() call targets sensitive path"
                                            f" '{first_arg.value}'"
                                        ),
                                    }

        return {"safe": True, "reason": None}

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """Validate *command* against the security policy then execute it.

        In ``mode="block"`` (default), any policy violation immediately raises
        `ValueError` and the code is never executed. In ``mode="warn"``, the
        violation is logged but execution proceeds.

        Args:
            command: Python source code to execute.
            timeout: Per-call timeout override in seconds. Falls back to the
                instance-level ``timeout`` attribute when not supplied.

        Returns:
            The captured stdout produced by the command.

        Raises:
            ValueError: When a policy violation is detected and ``mode="block"``.
        """
        effective_timeout = timeout if timeout is not None else self.timeout

        sanitized = self.sanitize_input(command)
        result = self._validate_code_safety(sanitized)

        if not result["safe"]:
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            snippet = sanitized[:200].replace("\n", "\\n")
            log_msg = (
                f"[SafePythonREPL] Policy violation at {timestamp} — "
                f"reason={result['reason']!r}, snippet={snippet!r}"
            )

            if self.mode == "block":
                logger.warning(log_msg)
                msg = f"Code blocked by security policy: {result['reason']}"
                raise ValueError(msg)

            # mode == "warn": UNSAFE — execution continues despite violation.
            # Development/testing environments only.
            logger.warning(log_msg)
            return super().run(command, timeout=effective_timeout)

        return super().run(command, timeout=effective_timeout)
