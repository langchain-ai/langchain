"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID

from langchainplus_sdk import LangChainPlusClient

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import (
    Run,
    RunTypeEnum,
    TracerSession,
)
from langchain.env import get_runtime_environment
from langchain.schema import BaseMessage, messages_to_dict

logger = logging.getLogger(__name__)
_LOGGED = set()
_TRACERS: List[LangChainTracer] = []


def log_error_once(method: str, exception: Exception) -> None:
    """Log an error once."""
    global _LOGGED
    if (method, type(exception)) in _LOGGED:
        return
    _LOGGED.add((method, type(exception)))
    logger.error(exception)


def wait_for_all_tracers() -> None:
    global _TRACERS
    for tracer in _TRACERS:
        tracer.wait_for_futures()


class LangChainTracer(BaseTracer):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    def __init__(
        self,
        example_id: Optional[Union[UUID, str]] = None,
        project_name: Optional[str] = None,
        client: Optional[LangChainPlusClient] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangChain tracer."""
        super().__init__(**kwargs)
        self.session: Optional[TracerSession] = None
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.project_name = project_name or os.getenv(
            "LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_SESSION", "default")
        )
        # set max_workers to 1 to process tasks in order
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.client = client or LangChainPlusClient()
        self._futures: Set[Future] = set()
        global _TRACERS
        _TRACERS.append(self)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for an LLM run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        chat_model_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [messages_to_dict(batch) for batch in messages]},
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=execution_order,
            child_execution_order=execution_order,
            run_type=RunTypeEnum.llm,
            tags=tags,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)

    def _persist_run(self, run: Run) -> None:
        """The Langchain Tracer uses Post/Patch rather than persist."""

    def _persist_run_single(self, run: Run) -> None:
        """Persist a run."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        run_dict = run.dict(exclude={"child_runs"})
        extra = run_dict.get("extra", {})
        extra["runtime"] = get_runtime_environment()
        run_dict["extra"] = extra
        try:
            self.client.create_run(**run_dict, project_name=self.project_name)
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            log_error_once("post", e)
            raise

    def _update_run_single(self, run: Run) -> None:
        """Update a run."""
        try:
            self.client.update_run(run.id, **run.dict())
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            log_error_once("patch", e)
            raise

    def _on_llm_start(self, run: Run) -> None:
        """Persist an LLM run."""
        self._futures.add(
            self.executor.submit(self._persist_run_single, run.copy(deep=True))
        )

    def _on_chat_model_start(self, run: Run) -> None:
        """Persist an LLM run."""
        self._futures.add(
            self.executor.submit(self._persist_run_single, run.copy(deep=True))
        )

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""
        self._futures.add(
            self.executor.submit(self._update_run_single, run.copy(deep=True))
        )

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""
        self._futures.add(
            self.executor.submit(self._update_run_single, run.copy(deep=True))
        )

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""
        self._futures.add(
            self.executor.submit(self._persist_run_single, run.copy(deep=True))
        )

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""
        self._futures.add(
            self.executor.submit(self._update_run_single, run.copy(deep=True))
        )

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""
        self._futures.add(
            self.executor.submit(self._update_run_single, run.copy(deep=True))
        )

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""
        self._futures.add(
            self.executor.submit(self._persist_run_single, run.copy(deep=True))
        )

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""
        self._futures.add(
            self.executor.submit(self._update_run_single, run.copy(deep=True))
        )

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""
        self._futures.add(
            self.executor.submit(self._update_run_single, run.copy(deep=True))
        )

    def wait_for_futures(self) -> None:
        """Wait for the given futures to complete."""
        futures = list(self._futures)
        wait(futures)
        for future in futures:
            self._futures.remove(future)
