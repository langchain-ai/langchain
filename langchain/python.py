"""Mock Python REPL."""
from typing import Dict, Optional


class PythonREPL:
    """Simulates a standalone Python REPL."""

    def __init__(self, _globals: Optional[Dict] = None, _locals: Optional[Dict] = None):
        """Initialize with optional globals and locals."""
        self._globals = _globals if _globals is not None else {}
        self._locals = _locals if _locals is not None else {}

    def run(self, command: str) -> None:
        """Run command with own globals/locals."""
        exec(command, self._globals, self._locals)
