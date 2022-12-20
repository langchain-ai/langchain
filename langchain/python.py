"""Mock Python REPL."""
import sys
from io import StringIO
from typing import Dict, Optional


class PythonREPL:
    """Simulates a standalone Python REPL."""

    def __init__(self, _globals: Optional[Dict] = None, _locals: Optional[Dict] = None):
        """Initialize with optional globals and locals."""
        self._globals = _globals if _globals is not None else {}
        self._locals = _locals if _locals is not None else {}

    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, self._globals, self._locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        return output
