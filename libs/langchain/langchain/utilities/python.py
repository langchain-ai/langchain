from __future__ import annotations

import functools
import logging
import multiprocessing
from typing import TYPE_CHECKING, Dict, Optional

from langchain.pydantic_v1 import BaseModel, Field

if TYPE_CHECKING:
    from wasm_exec import WasmExecutor

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


def _get_wasm_executor() -> WasmExecutor:
    try:
        from wasm_exec import WasmExecutor
    except ImportError as e:
        raise ImportError(
            "Unable to import wasm_exec, please install with `pip install wasm_exec`."
        ) from e

    return WasmExecutor()


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")
    wasm: WasmExecutor = Field(default_factory=_get_wasm_executor)

    @classmethod
    def worker(
        cls,
        command: str,
        globals: Optional[Dict],
        locals: Optional[Dict],
        wasm: WasmExecutor,
        queue: multiprocessing.Queue,
    ) -> None:
        try:
            result = wasm.exec(command, globals=globals, locals=locals)
            queue.put(result.text)
        except Exception as e:
            queue.put(repr(e))

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""

        # Warn against dangers of PythonREPL
        warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, self.locals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return "Execution timed out"
        else:
            self.worker(command, self.globals, self.locals, self.wasm, queue)
        # get the result from the worker function
        return queue.get()
