"""A Tracer implementation that records to LangChain endpoint."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langsmith import Client, get_tracing_context
from langsmith import run_trees as rt
from langsmith import utils as ls_utils
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import override

from langchain_core.env import get_runtime_environment
from langchain_core.load import dumpd
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

logger = logging.getLogger(__name__)
_LOGGED = set()
_EXECUTOR: ThreadPoolExecutor | None = None


def log_error_once(method: str, exception: Exception) -> None:
    """Log an error once.

    Args:
        method: The method that raised the exception.
        exception: The exception that was raised.
    """
    if (method, type(exception)) in _LOGGED:
        return
    _LOGGED.add((method, type(exception)))
    logger.error(exception)


def wait_for_all_tracers() -> None:
    """Wait for all tracers to finish."""
    if rt._CLIENT is not None:  # noqa: SLF001
        rt._CLIENT.flush()  # noqa: SLF001


def get_client() -> Client:
    """Get the client.

    Returns:
        The LangSmith client.
    """
    return rt.get_cached_client()


def _get_executor() -> ThreadPoolExecutor:
    """Get the executor."""
    global _EXECUTOR  # noqa: PLW0603
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor()
    return _EXECUTOR


def _get_usage_metadata_from_generations(
    generations: list[list[dict[str, Any]]],
) -> UsageMetadata | None:
    """Extract and aggregate `usage_metadata` from generations.

    Iterates through generations to find and aggregate all `usage_metadata` found in
    messages. This is typically present in chat model outputs.

    Args:
        generations: List of generation batches, where each batch is a list
            of generation dicts that may contain a `'message'` key with
            `'usage_metadata'`.

    Returns:
        The aggregated `usage_metadata` dict if found, otherwise `None`.
    """
    output: UsageMetadata | None = None
    for generation_batch in generations:
        for generation in generation_batch:
            if isinstance(generation, dict) and "message" in generation:
                message = generation["message"]
                if isinstance(message, dict) and "usage_metadata" in message:
                    output = add_usage(output, message["usage_metadata"])
    return output


class LangChainTracer(BaseTracer):
    """Implementation of the SharedTracer that POSTS to the LangChain endpoint."""

    run_inline = True

    def __init__(
        self,
        example_id: UUID | str | None = None,
        project_name: str | None = None,
        client: Client | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangChain tracer.

        Args:
            example_id: The example ID.
            project_name: The project name. Defaults to the tracer project.
            client: The client. Defaults to the global client.
            tags: The tags. Defaults to an empty list.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.project_name = project_name or ls_utils.get_tracer_project()
        self.client = client or get_client()
        self.tags = tags or []
        self.latest_run: Run | None = None
        self.run_has_token_event_map: dict[str, bool] = {}

    def _start_trace(self, run: Run) -> None:
        if self.project_name:
            run.session_name = self.project_name
        if self.tags is not None:
            if run.tags:
                run.tags = sorted(set(run.tags + self.tags))
            else:
                run.tags = self.tags.copy()

        super()._start_trace(run)
        if run.ls_client is None:
            run.ls_client = self.client
        if get_tracing_context().get("enabled") is False:
            run.extra["__disabled"] = True

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for an LLM run.

        Args:
            serialized: The serialized model.
            messages: The messages.
            run_id: The run ID.
            tags: The tags.
            parent_run_id: The parent run ID.
            metadata: The metadata.
            name: The name.
            **kwargs: Additional keyword arguments.

        Returns:
            The run.
        """
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        chat_model_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [[dumpd(msg) for msg in batch] for batch in messages]},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            run_type="llm",
            tags=tags,
            name=name,  # type: ignore[arg-type]
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def _persist_run(self, run: Run) -> None:
        # We want to free up more memory by avoiding keeping a reference to the
        # whole nested run tree.
        self.latest_run = Run.construct(
            **run.dict(exclude={"child_runs", "inputs", "outputs"}),
            inputs=run.inputs,
            outputs=run.outputs,
        )

    def get_run_url(self) -> str:
        """Get the LangSmith root run URL.

        Returns:
            The LangSmith root run URL.

        Raises:
            ValueError: If no traced run is found.
            ValueError: If the run URL cannot be found.
        """
        if not self.latest_run:
            msg = "No traced run found."
            raise ValueError(msg)
        # If this is the first run in a project, the project may not yet be created.
        # This method is only really useful for debugging flows, so we will assume
        # there is some tolerace for latency.
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential_jitter(),
            retry=retry_if_exception_type(ls_utils.LangSmithError),
        ):
            with attempt:
                return self.client.get_run_url(
                    run=self.latest_run, project_name=self.project_name
                )
        msg = "Failed to get run URL."
        raise ValueError(msg)

    def _get_tags(self, run: Run) -> list[str]:
        """Get combined tags for a run."""
        tags = set(run.tags or [])
        tags.update(self.tags or [])
        return list(tags)

    def _persist_run_single(self, run: Run) -> None:
        """Persist a run."""
        if run.extra.get("__disabled"):
            return
        try:
            run.extra["runtime"] = get_runtime_environment()
            run.tags = self._get_tags(run)
            if run.ls_client is not self.client:
                run.ls_client = self.client
            run.post()
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            log_error_once("post", e)
            raise

    @staticmethod
    def _update_run_single(run: Run) -> None:
        """Update a run."""
        if run.extra.get("__disabled"):
            return
        try:
            run.patch(exclude_inputs=run.extra.get("inputs_is_truthy", False))
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            log_error_once("patch", e)
            raise

    def _on_llm_start(self, run: Run) -> None:
        """Persist an LLM run."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    @override
    def _llm_run_with_token_event(
        self,
        token: str,
        run_id: UUID,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        parent_run_id: UUID | None = None,
    ) -> Run:
        run_id_str = str(run_id)
        if run_id_str not in self.run_has_token_event_map:
            self.run_has_token_event_map[run_id_str] = True
        else:
            return self._get_run(run_id, run_type={"llm", "chat_model"})
        return super()._llm_run_with_token_event(
            # Drop the chunk; we don't need to save it
            token,
            run_id,
            chunk=None,
            parent_run_id=parent_run_id,
        )

    def _on_chat_model_start(self, run: Run) -> None:
        """Persist an LLM run."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""
        # Extract usage_metadata from outputs and store in extra.metadata
        if run.outputs and "generations" in run.outputs:
            usage_metadata = _get_usage_metadata_from_generations(
                run.outputs["generations"]
            )
            if usage_metadata is not None:
                if "metadata" not in run.extra:
                    run.extra["metadata"] = {}
                run.extra["metadata"]["usage_metadata"] = usage_metadata
        self._update_run_single(run)

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""
        self._update_run_single(run)

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        # Skip persisting if inputs are deferred (e.g., iterator/generator inputs).
        # The run will be posted when _on_chain_end is called with realized inputs.
        if not run.extra.get("defers_inputs"):
            self._persist_run_single(run)

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""
        # If inputs were deferred, persist (POST) the run now that inputs are realized.
        # Otherwise, update (PATCH) the existing run.
        if run.extra.get("defers_inputs"):
            self._persist_run_single(run)
        else:
            self._update_run_single(run)

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""
        # If inputs were deferred, persist (POST) the run now that inputs are realized.
        # Otherwise, update (PATCH) the existing run.
        if run.extra.get("defers_inputs"):
            self._persist_run_single(run)
        else:
            self._update_run_single(run)

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""
        self._update_run_single(run)

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""
        self._update_run_single(run)

    def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run."""
        self._update_run_single(run)

    def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error."""
        self._update_run_single(run)

    def wait_for_futures(self) -> None:
        """Wait for the given futures to complete."""
        if self.client is not None:
            self.client.flush()
