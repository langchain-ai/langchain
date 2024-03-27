from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import requests

from langchain_core._api import deprecated
from langchain_core.messages import get_buffer_string
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import (
    ChainRun,
    LLMRun,
    Run,
    ToolRun,
    TracerSession,
    TracerSessionV1,
    TracerSessionV1Base,
)
from langchain_core.utils import raise_for_status_with_text

logger = logging.getLogger(__name__)


def get_headers() -> Dict[str, Any]:
    """Get the headers for the LangChain API."""
    headers: Dict[str, Any] = {"Content-Type": "application/json"}
    if os.getenv("LANGCHAIN_API_KEY"):
        headers["x-api-key"] = os.getenv("LANGCHAIN_API_KEY")
    return headers


def _get_endpoint() -> str:
    return os.getenv("LANGCHAIN_ENDPOINT", "http://localhost:8000")


@deprecated("0.1.0", alternative="LangChainTracer", removal="0.2.0")
class LangChainTracerV1(BaseTracer):
    """Implementation of the SharedTracer that POSTS to the langchain endpoint."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LangChain tracer."""
        super().__init__(**kwargs)
        self.session: Optional[TracerSessionV1] = None
        self._endpoint = _get_endpoint()
        self._headers = get_headers()

    def _convert_to_v1_run(self, run: Run) -> Union[LLMRun, ChainRun, ToolRun]:
        session = self.session or self.load_default_session()
        if not isinstance(session, TracerSessionV1):
            raise ValueError(
                "LangChainTracerV1 is not compatible with"
                f" session of type {type(session)}"
            )

        if run.run_type == "llm":
            if "prompts" in run.inputs:
                prompts = run.inputs["prompts"]
            elif "messages" in run.inputs:
                prompts = [get_buffer_string(batch) for batch in run.inputs["messages"]]
            else:
                raise ValueError("No prompts found in LLM run inputs")
            return LLMRun(
                uuid=str(run.id) if run.id else None,  # type: ignore[arg-type]
                parent_uuid=str(run.parent_run_id) if run.parent_run_id else None,
                start_time=run.start_time,
                end_time=run.end_time,  # type: ignore[arg-type]
                extra=run.extra,
                execution_order=run.execution_order,
                child_execution_order=run.child_execution_order,
                serialized=run.serialized,  # type: ignore[arg-type]
                session_id=session.id,
                error=run.error,
                prompts=prompts,
                response=run.outputs if run.outputs else None,  # type: ignore[arg-type]
            )
        if run.run_type == "chain":
            child_runs = [self._convert_to_v1_run(run) for run in run.child_runs]
            return ChainRun(
                uuid=str(run.id) if run.id else None,  # type: ignore[arg-type]
                parent_uuid=str(run.parent_run_id) if run.parent_run_id else None,
                start_time=run.start_time,
                end_time=run.end_time,  # type: ignore[arg-type]
                execution_order=run.execution_order,
                child_execution_order=run.child_execution_order,
                serialized=run.serialized,  # type: ignore[arg-type]
                session_id=session.id,
                inputs=run.inputs,
                outputs=run.outputs,
                error=run.error,
                extra=run.extra,
                child_llm_runs=[run for run in child_runs if isinstance(run, LLMRun)],
                child_chain_runs=[
                    run for run in child_runs if isinstance(run, ChainRun)
                ],
                child_tool_runs=[run for run in child_runs if isinstance(run, ToolRun)],
            )
        if run.run_type == "tool":
            child_runs = [self._convert_to_v1_run(run) for run in run.child_runs]
            return ToolRun(
                uuid=str(run.id) if run.id else None,  # type: ignore[arg-type]
                parent_uuid=str(run.parent_run_id) if run.parent_run_id else None,
                start_time=run.start_time,
                end_time=run.end_time,  # type: ignore[arg-type]
                execution_order=run.execution_order,
                child_execution_order=run.child_execution_order,
                serialized=run.serialized,  # type: ignore[arg-type]
                session_id=session.id,
                action=str(run.serialized),
                tool_input=run.inputs.get("input", ""),
                output=None if run.outputs is None else run.outputs.get("output"),
                error=run.error,
                extra=run.extra,
                child_chain_runs=[
                    run for run in child_runs if isinstance(run, ChainRun)
                ],
                child_tool_runs=[run for run in child_runs if isinstance(run, ToolRun)],
                child_llm_runs=[run for run in child_runs if isinstance(run, LLMRun)],
            )
        raise ValueError(f"Unknown run type: {run.run_type}")

    def _persist_run(self, run: Union[Run, LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        if isinstance(run, Run):
            v1_run = self._convert_to_v1_run(run)
        else:
            v1_run = run
        if isinstance(v1_run, LLMRun):
            endpoint = f"{self._endpoint}/llm-runs"
        elif isinstance(v1_run, ChainRun):
            endpoint = f"{self._endpoint}/chain-runs"
        else:
            endpoint = f"{self._endpoint}/tool-runs"

        try:
            response = requests.post(
                endpoint,
                data=v1_run.json(),
                headers=self._headers,
            )
            raise_for_status_with_text(response)
        except Exception as e:
            logger.warning(f"Failed to persist run: {e}")

    def _persist_session(
        self, session_create: TracerSessionV1Base
    ) -> Union[TracerSessionV1, TracerSession]:
        """Persist a session."""
        try:
            r = requests.post(
                f"{self._endpoint}/sessions",
                data=session_create.json(),
                headers=self._headers,
            )
            session = TracerSessionV1(id=r.json()["id"], **session_create.dict())
        except Exception as e:
            logger.warning(f"Failed to create session, using default session: {e}")
            session = TracerSessionV1(id=1, **session_create.dict())
        return session

    def _load_session(self, session_name: Optional[str] = None) -> TracerSessionV1:
        """Load a session from the tracer."""
        try:
            url = f"{self._endpoint}/sessions"
            if session_name:
                url += f"?name={session_name}"
            r = requests.get(url, headers=self._headers)

            tracer_session = TracerSessionV1(**r.json()[0])
        except Exception as e:
            session_type = "default" if not session_name else session_name
            logger.warning(
                f"Failed to load {session_type} session, using empty session: {e}"
            )
            tracer_session = TracerSessionV1(id=1)

        self.session = tracer_session
        return tracer_session

    def load_session(self, session_name: str) -> Union[TracerSessionV1, TracerSession]:
        """Load a session with the given name from the tracer."""
        return self._load_session(session_name)

    def load_default_session(self) -> Union[TracerSessionV1, TracerSession]:
        """Load the default tracing session and set it as the Tracer's session."""
        return self._load_session("default")
