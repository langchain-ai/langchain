"""Тут обновленная версия python.tools, которая возвращает результат кода +
информацию об ошибке, если она произошла во время выполнения"""

import functools
import logging
import multiprocessing
import sys
import traceback
from io import StringIO
from typing import Dict, Optional, TypedDict

from langchain.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


class CodeExecutionResult(TypedDict):
    result: str
    is_exception: bool
    is_timeout: bool


class GigaPythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")
    timeout: Optional[int] = None

    @classmethod
    def worker(
        cls,
        command: str,
        globals: Optional[Dict],
        locals: Optional[Dict],
        queue: multiprocessing.Queue,
    ) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, globals, locals)
            sys.stdout = old_stdout
            queue.put(
                CodeExecutionResult(
                    result=mystdout.getvalue(), is_exception=False, is_timeout=False
                )
            )
        except Exception as err:
            if issubclass(type(err), (SyntaxError,)):
                line_number = err.lineno  # type: ignore
            else:
                cl, exc, tb = sys.exc_info()
                tb = traceback.extract_tb(tb)  # type: ignore
                line_number = tb[1][1]  # type: ignore
            code_info = ""
            if command is not None:
                line = command.splitlines()[line_number - 1]
                code_info = f" -> {line}"
            sys.stdout = old_stdout
            queue.put(
                CodeExecutionResult(
                    result=repr(err) + code_info, is_exception=True, is_timeout=False
                )
            )

    def run(self, command: str) -> CodeExecutionResult:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""

        # Warn against dangers of PythonREPL
        warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if self.timeout is not None:
            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, self.locals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(self.timeout)

            if p.is_alive():
                p.terminate()
                return CodeExecutionResult(
                    is_exception=True, result="Execution timed out", is_timeout=True
                )
        else:
            self.worker(command, self.globals, self.locals, queue)
        # get the result from the worker function
        return queue.get()
