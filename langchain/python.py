"""Mock Python REPL."""
import sys
from io import StringIO
from typing import Dict, Optional

from pydantic import BaseModel, Field


def remove_enclosing_markdown_for_python(command: str) -> str:
    """
    Removes any enclosing markdown for python code: inline markdown,
    block markdown and formatted block code markdown
    """
    # Markdown with specified python
    if command.startswith("```python") and command.endswith("```"):
        command = command[9:-3]
    elif command.startswith("```") and command.endswith("```") and len(command) > 6:
        command = command[3:-3]
    elif command.startswith("`") and command.endswith("`") and len(command) > 2:
        command = command[1:-1]

    # New lines can cause syntax errors
    return command.strip()


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        # Ignore any enclosing markdown, ChatGPT does this
        command = remove_enclosing_markdown_for_python(command)

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
