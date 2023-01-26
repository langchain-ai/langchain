"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations

import logging
import os
from abc import ABC
from typing import Any, Dict, Optional, Union

import requests

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import (
    ChainRun,
    LLMRun,
    ToolRun,
    TracerSession,
    TracerSessionCreate,
)


class BaseLangChainTracer(BaseTracer, ABC):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    always_verbose: bool = True
    _endpoint: str = os.getenv("LANGCHAIN_ENDPOINT", "http://localhost:8000")
    _headers: Dict[str, Any] = {"Content-Type": "application/json"}
    if os.getenv("LANGCHAIN_API_KEY"):
        _headers["x-api-key"] = os.getenv("LANGCHAIN_API_KEY")

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        if isinstance(run, LLMRun):
            endpoint = f"{self._endpoint}/llm-runs"
        elif isinstance(run, ChainRun):
            endpoint = f"{self._endpoint}/chain-runs"
        else:
            endpoint = f"{self._endpoint}/tool-runs"

        try:
            requests.post(
                endpoint,
                data=run.json(),
                headers=self._headers,
            )
        except Exception as e:
            logging.warning(f"Failed to persist run: {e}")

    def _persist_session(self, session_create: TracerSessionCreate) -> TracerSession:
        """Persist a session."""
        try:
            r = requests.post(
                f"{self._endpoint}/sessions",
                data=session_create.json(),
                headers=self._headers,
            )
            session = TracerSession(id=r.json()["id"], **session_create.dict())
        except Exception as e:
            logging.warning(f"Failed to create session, using default session: {e}")
            session = TracerSession(id=1, **session_create.dict())
        return session

    def load_session(self, session_name: str) -> TracerSession:
        """Load a session from the tracer."""
        try:
            r = requests.get(
                f"{self._endpoint}/sessions?name={session_name}",
                headers=self._headers,
            )
            tracer_session = TracerSession(**r.json()[0])
            self._session = tracer_session
            return tracer_session
        except Exception as e:
            logging.warning(
                f"Failed to load session {session_name}, using empty session: {e}"
            )
            tracer_session = TracerSession(id=1)
            self._session = tracer_session
            return tracer_session

    def load_default_session(self) -> TracerSession:
        """Load the default tracing session and set it as the Tracer's session."""
        try:
            r = requests.get(
                f"{self._endpoint}/sessions",
                headers=self._headers,
            )
            # Use the first session result
            tracer_session = TracerSession(**r.json()[0])
            self._session = tracer_session
            return tracer_session
        except Exception as e:
            logging.warning(f"Failed to default session, using empty session: {e}")
            tracer_session = TracerSession(id=1)
            self._session = tracer_session
            return tracer_session

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""
        if isinstance(child_run, LLMRun):
            parent_run.child_llm_runs.append(child_run)
        elif isinstance(child_run, ChainRun):
            parent_run.child_chain_runs.append(child_run)
        else:
            parent_run.child_tool_runs.append(child_run)

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""
        return None
