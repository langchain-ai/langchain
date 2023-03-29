"""Mock Python REPL."""
import sys
from io import StringIO
from typing import Dict, Optional

from pydantic import BaseModel, Field


def remove_backticks_if_they_exist(command: str) -> str:
    if (len(command) >= 6 and command[:3] == "```" and command[:3] == "```"):
        command = command[3:-3]
    return command


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    
    
    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        # Ignore any backticks if provided by chat models, ChatGPT does this
        command = remove_backticks_if_they_exist(command)
            
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, self.globals, self.locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        return output
