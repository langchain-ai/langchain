"""Unit tests for SafePythonREPL AST-based security filtering."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain.utilities.python import (
    DEFAULT_DENIED_FUNCTIONS,
    DEFAULT_DENIED_IMPORTS,
    PythonREPL,
    SafePythonREPL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repl(**kwargs: Any) -> SafePythonREPL:
    """Construct a SafePythonREPL, bypassing multiprocessing for unit tests."""
    return SafePythonREPL(**kwargs)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------


class TestDefaultBlocklists:
    def test_default_denied_imports_contains_os(self) -> None:
        assert "os" in DEFAULT_DENIED_IMPORTS

    def test_default_denied_imports_contains_subprocess(self) -> None:
        assert "subprocess" in DEFAULT_DENIED_IMPORTS

    def test_default_denied_functions_contains_eval(self) -> None:
        assert "eval" in DEFAULT_DENIED_FUNCTIONS

    def test_default_denied_functions_contains_exec(self) -> None:
        assert "exec" in DEFAULT_DENIED_FUNCTIONS


# ---------------------------------------------------------------------------
# Mode parameter
# ---------------------------------------------------------------------------


class TestModeDefault:
    def test_default_mode_is_block(self) -> None:
        repl = _make_repl()
        assert repl.mode == "block"

    def test_explicit_block_mode(self) -> None:
        repl = _make_repl(mode="block")
        assert repl.mode == "block"

    def test_explicit_warn_mode(self) -> None:
        repl = _make_repl(mode="warn")
        assert repl.mode == "warn"


# ---------------------------------------------------------------------------
# AST validation — _validate_code_safety
# ---------------------------------------------------------------------------


class TestValidateCodeSafety:
    """Direct tests of the AST validator without touching exec()."""

    def _repl(self, **kwargs: Any) -> SafePythonREPL:
        return _make_repl(**kwargs)

    # Imports

    def test_blocks_os_import(self) -> None:
        result = self._repl()._validate_code_safety("import os")
        assert result["safe"] is False
        assert "os" in result["reason"]

    def test_blocks_subprocess_import(self) -> None:
        result = self._repl()._validate_code_safety("import subprocess")
        assert result["safe"] is False

    def test_blocks_sys_import(self) -> None:
        result = self._repl()._validate_code_safety("import sys")
        assert result["safe"] is False

    def test_blocks_socket_import(self) -> None:
        result = self._repl()._validate_code_safety("import socket")
        assert result["safe"] is False

    def test_blocks_urllib_import(self) -> None:
        result = self._repl()._validate_code_safety("import urllib")
        assert result["safe"] is False

    def test_blocks_requests_import(self) -> None:
        result = self._repl()._validate_code_safety("import requests")
        assert result["safe"] is False

    def test_blocks_from_os_import(self) -> None:
        result = self._repl()._validate_code_safety("from os import path")
        assert result["safe"] is False
        assert "os" in result["reason"]

    def test_blocks_nested_from_import(self) -> None:
        # from os.path import join — root module 'os' is denied
        result = self._repl()._validate_code_safety("from os.path import join")
        assert result["safe"] is False

    def test_blocks_dotted_import(self) -> None:
        # import os.path — root module 'os' is denied
        result = self._repl()._validate_code_safety("import os.path")
        assert result["safe"] is False

    # Denied functions

    def test_blocks_eval_call(self) -> None:
        result = self._repl()._validate_code_safety("eval('1+1')")
        assert result["safe"] is False
        assert "eval" in result["reason"]

    def test_blocks_exec_call(self) -> None:
        result = self._repl()._validate_code_safety("exec('x=1')")
        assert result["safe"] is False

    def test_blocks_dunder_import(self) -> None:
        result = self._repl()._validate_code_safety("__import__('os')")
        assert result["safe"] is False

    def test_blocks_compile_call(self) -> None:
        result = self._repl()._validate_code_safety("compile('x=1', '', 'exec')")
        assert result["safe"] is False

    def test_blocks_open_call(self) -> None:
        result = self._repl()._validate_code_safety("open('data.csv')")
        assert result["safe"] is False

    def test_blocks_globals_call(self) -> None:
        result = self._repl()._validate_code_safety("globals()")
        assert result["safe"] is False

    def test_blocks_locals_call(self) -> None:
        result = self._repl()._validate_code_safety("locals()")
        assert result["safe"] is False

    def test_blocks_vars_call(self) -> None:
        result = self._repl()._validate_code_safety("vars()")
        assert result["safe"] is False

    def test_blocks_input_call(self) -> None:
        result = self._repl()._validate_code_safety("input('prompt')")
        assert result["safe"] is False

    # Sensitive paths

    def test_blocks_open_etc_passwd(self) -> None:
        # open() is in DEFAULT_DENIED_FUNCTIONS so it is blocked by the function
        # check; the sensitive-path guard provides an additional layer when open()
        # has been explicitly allowed via a custom denied_functions set.
        result = self._repl()._validate_code_safety("open('/etc/passwd')")
        assert result["safe"] is False

    def test_blocks_open_proc_path(self) -> None:
        result = self._repl()._validate_code_safety("open('/proc/cpuinfo')")
        assert result["safe"] is False

    def test_blocks_open_sys_path(self) -> None:
        result = self._repl()._validate_code_safety("open('/sys/kernel/debug')")
        assert result["safe"] is False

    def test_sensitive_path_guard_triggers_when_open_not_in_denied_functions(
        self,
    ) -> None:
        """Sensitive-path check fires even when 'open' is removed from denied_functions."""
        repl = _make_repl(denied_functions=set())  # no functions blocked
        result = repl._validate_code_safety("open('/etc/passwd')")
        assert result["safe"] is False
        assert "/etc/passwd" in result["reason"]

    # Attribute access on denied modules

    def test_blocks_attribute_access_on_denied_module(self) -> None:
        # socket.connect() where 'socket' is in the denied imports
        result = self._repl()._validate_code_safety("socket.connect('127.0.0.1')")
        assert result["safe"] is False

    # Safe code

    def test_allows_basic_arithmetic(self) -> None:
        result = self._repl()._validate_code_safety("x = 2 + 2")
        assert result["safe"] is True
        assert result["reason"] is None

    def test_allows_string_operations(self) -> None:
        result = self._repl()._validate_code_safety("s = 'hello'.upper()")
        assert result["safe"] is True

    def test_allows_list_comprehension(self) -> None:
        result = self._repl()._validate_code_safety(
            "squares = [x**2 for x in range(10)]"
        )
        assert result["safe"] is True

    def test_allows_function_definition(self) -> None:
        result = self._repl()._validate_code_safety(
            "def add(a, b):\n    return a + b"
        )
        assert result["safe"] is True

    def test_allows_print(self) -> None:
        result = self._repl()._validate_code_safety("print('hello')")
        assert result["safe"] is True

    def test_allows_math_import(self) -> None:
        result = self._repl()._validate_code_safety("import math\nprint(math.pi)")
        assert result["safe"] is True

    def test_allows_json_import(self) -> None:
        result = self._repl()._validate_code_safety(
            "import json\njson.dumps({'a': 1})"
        )
        assert result["safe"] is True

    def test_syntax_error_is_treated_as_safe(self) -> None:
        # Unparseable code is not flagged — exec() will surface the SyntaxError.
        result = self._repl()._validate_code_safety("def broken(")
        assert result["safe"] is True

    # Custom allowlists

    def test_allowed_imports_overrides_denied(self) -> None:
        repl = _make_repl(allowed_imports={"os"})
        result = repl._validate_code_safety("import os")
        assert result["safe"] is True

    def test_allowed_imports_only_overrides_named_module(self) -> None:
        repl = _make_repl(allowed_imports={"os"})
        result = repl._validate_code_safety("import subprocess")
        assert result["safe"] is False

    def test_custom_denied_imports_replaces_defaults(self) -> None:
        # Use a custom set that doesn't include 'os' but adds 'mylib'
        repl = _make_repl(denied_imports={"mylib"})
        assert repl._validate_code_safety("import os")["safe"] is True
        assert repl._validate_code_safety("import mylib")["safe"] is False

    def test_custom_denied_functions_replaces_defaults(self) -> None:
        repl = _make_repl(denied_functions={"dangerous_fn"})
        assert repl._validate_code_safety("eval('x')")["safe"] is True
        assert repl._validate_code_safety("dangerous_fn()")["safe"] is False

    def test_empty_allowed_imports_has_no_effect(self) -> None:
        repl = _make_repl(allowed_imports=set())
        assert repl._validate_code_safety("import os")["safe"] is False

    # Edge cases

    def test_nested_import_root_blocked(self) -> None:
        result = self._repl()._validate_code_safety("import urllib.request")
        assert result["safe"] is False

    def test_empty_code_is_safe(self) -> None:
        result = self._repl()._validate_code_safety("")
        assert result["safe"] is True

    def test_comment_only_is_safe(self) -> None:
        result = self._repl()._validate_code_safety("# just a comment")
        assert result["safe"] is True

    def test_multiline_safe_code(self) -> None:
        code = "x = 1\ny = 2\nz = x + y\nprint(z)"
        result = self._repl()._validate_code_safety(code)
        assert result["safe"] is True


# ---------------------------------------------------------------------------
# run() — block mode
# ---------------------------------------------------------------------------


class TestRunBlockMode:
    """Ensure that run() raises ValueError before exec() is called."""

    def test_raises_on_os_import(self) -> None:
        repl = _make_repl()
        with pytest.raises(ValueError, match="security policy"):
            repl.run("import os")

    def test_raises_on_subprocess_import(self) -> None:
        repl = _make_repl()
        with pytest.raises(ValueError):
            repl.run("import subprocess")

    def test_raises_on_eval(self) -> None:
        repl = _make_repl()
        with pytest.raises(ValueError, match="eval"):
            repl.run("eval('1+1')")

    def test_raises_on_exec(self) -> None:
        repl = _make_repl()
        with pytest.raises(ValueError):
            repl.run("exec('x=1')")

    def test_raises_on_open_sensitive_path(self) -> None:
        repl = _make_repl()
        with pytest.raises(ValueError, match="security policy"):
            repl.run("open('/etc/passwd')")

    def test_error_message_mentions_policy(self) -> None:
        repl = _make_repl()
        with pytest.raises(ValueError) as exc_info:
            repl.run("import os")
        assert "security policy" in str(exc_info.value).lower()

    def test_error_message_does_not_echo_full_code(self) -> None:
        """Violation messages must not leak the raw code back to the caller."""
        repl = _make_repl()
        secret = "import os; os.getenv('SECRET_API_KEY')"
        with pytest.raises(ValueError) as exc_info:
            repl.run(secret)
        assert secret not in str(exc_info.value)

    def test_block_mode_does_not_execute(self) -> None:
        """Verify that exec() is never reached when a violation is detected."""
        repl = _make_repl(mode="block")
        with patch.object(PythonREPL, "worker") as mock_worker:
            with pytest.raises(ValueError):
                repl.run("import os")
            mock_worker.assert_not_called()

    def test_violation_logged_in_block_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        repl = _make_repl(mode="block")
        with caplog.at_level(logging.WARNING, logger="langchain.utilities.python"):
            with pytest.raises(ValueError):
                repl.run("import os")
        assert any("Policy violation" in r.message for r in caplog.records)

    def test_safe_code_does_not_raise(self) -> None:
        repl = _make_repl(mode="block")
        with patch.object(PythonREPL, "run", return_value="4\n"):
            result = repl.run("print(2 + 2)")
        assert result == "4\n"


# ---------------------------------------------------------------------------
# run() — warn mode
# ---------------------------------------------------------------------------


class TestRunWarnMode:
    def test_warn_mode_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        repl = _make_repl(mode="warn")
        with caplog.at_level(logging.WARNING, logger="langchain.utilities.python"):
            with patch.object(PythonREPL, "run", return_value=""):
                repl.run("import os")
        # No exception raised — just a warning logged.

    def test_warn_mode_logs_violation(self, caplog: pytest.LogCaptureFixture) -> None:
        repl = _make_repl(mode="warn")
        with caplog.at_level(logging.WARNING, logger="langchain.utilities.python"):
            with patch.object(PythonREPL, "run", return_value=""):
                repl.run("import os")
        assert any("Policy violation" in r.message for r in caplog.records)

    def test_warn_mode_log_contains_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        repl = _make_repl(mode="warn")
        with caplog.at_level(logging.WARNING, logger="langchain.utilities.python"):
            with patch.object(PythonREPL, "run", return_value=""):
                repl.run("import os")
        combined = " ".join(r.message for r in caplog.records)
        assert "os" in combined

    def test_warn_mode_log_contains_timestamp(self, caplog: pytest.LogCaptureFixture) -> None:
        repl = _make_repl(mode="warn")
        with caplog.at_level(logging.WARNING, logger="langchain.utilities.python"):
            with patch.object(PythonREPL, "run", return_value=""):
                repl.run("eval('x')")
        combined = " ".join(r.message for r in caplog.records)
        # ISO timestamp contains a 'T' separator between date and time.
        assert "T" in combined

    def test_warn_mode_calls_parent_run(self) -> None:
        repl = _make_repl(mode="warn")
        mock_run = MagicMock(return_value="output")
        with patch.object(PythonREPL, "run", mock_run):
            repl.run("import os")
        mock_run.assert_called_once()

    def test_warn_mode_safe_code_executes(self) -> None:
        repl = _make_repl(mode="warn")
        with patch.object(PythonREPL, "run", return_value="9\n"):
            result = repl.run("print(3 ** 2)")
        assert result == "9\n"


# ---------------------------------------------------------------------------
# Backward compatibility — PythonREPL is unchanged
# ---------------------------------------------------------------------------


class TestPythonREPLUnchanged:
    def test_python_repl_instantiates(self) -> None:
        repl = PythonREPL()
        assert repl is not None

    def test_python_repl_sanitize_input(self) -> None:
        assert PythonREPL.sanitize_input("```python\nprint(1)\n```") == "print(1)"

    def test_python_repl_has_no_mode_attribute(self) -> None:
        repl = PythonREPL()
        assert not hasattr(repl, "mode")

    def test_safe_repl_is_subclass(self) -> None:
        assert issubclass(SafePythonREPL, PythonREPL)
