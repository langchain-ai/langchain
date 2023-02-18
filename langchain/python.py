"""Mock Python REPL."""
import sys
from io import StringIO
from typing import Dict, Optional
from RestrictedPython import compile_restricted, utility_builtins, safe_builtins, limited_builtins
from RestrictedPython.PrintCollector import PrintCollector
_print_ = PrintCollector
_getattr_ = getattr

def default_guarded_getitem(ob, index):
    return ob[index]

class PythonREPL:
    """Simulates a standalone Python REPL."""

    def __init__(self, _globals: Optional[Dict] = None, _locals: Optional[Dict] = None):
        """Initialize with optional globals and locals."""
        self._globals = _globals if _globals is not None else {}
        self._locals = _locals if _locals is not None else {}


    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        # old_stdout = sys.stdout
        # sys.stdout = mystdout = StringIO()
        try:
            _print_ = PrintCollector
            _getattr_ = getattr

            globals = {
                '__builtins__': safe_builtins,
                "str": str,
                "dict": dict,
                "_print_": _print_,
                "_getattr_": _getattr_,
                "_getitem_": default_guarded_getitem,
                }
            globals = {**globals, **self._globals}

            locals = {**{"result": None}, **self._locals}

            byte_code = compile_restricted(command, filename='<inline code>', mode='exec')

            exec(byte_code, globals, locals)

            output = str(locals["result"])
            # exec(command, self._globals, self._locals)
            # sys.stdout = old_stdout
            # output = mystdout.getvalue()
        except Exception as e:
            # sys.stdout = old_stdout
            output = str(e)
        return output
