"""A simple progress bar for the console."""
import threading
from typing import Any, Dict, Optional, Sequence
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from langchain.callbacks import base as base_callbacks


class ProgressBarCallback(base_callbacks.BaseCallbackHandler):
    """A simple progress bar for the console."""

    def __init__(self, total: int, ncols: int = 50, **kwargs: Any):
        """Initialize the progress bar.

        Args:
            total: int, the total number of items to be processed.
            ncols: int, the character width of the progress bar.
        """
        self.total = total
        self.ncols = ncols
        self.counter = 0
        self.lock = threading.Lock()
        self._print_bar()

    def increment(self) -> None:
        """Increment the counter and update the progress bar."""
        with self.lock:
            self.counter += 1
            self._print_bar()

    def _print_bar(self) -> None:
        """Print the progress bar to the console."""
        progress = self.counter / self.total
        arrow = "-" * int(round(progress * self.ncols) - 1) + ">"
        spaces = " " * (self.ncols - len(arrow))
        print(f"\r[{arrow + spaces}] {self.counter}/{self.total}", end="")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()
