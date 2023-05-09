"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import requests

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import (
    ChainRun,
    LLMRun,
    RunCreate,
    ToolRun,
    TracerSession,
    TracerSessionBase,
    TracerSessionV2,
    TracerSessionV2Create,
)
from langchain.utils import raise_for_status_with_text


def _get_headers() -> Dict[str, Any]:
    """Get the headers for the LangChain API."""
    headers: Dict[str, Any] = {"Content-Type": "application/json"}
    if os.getenv("LANGCHAIN_API_KEY"):
        headers["x-api-key"] = os.getenv("LANGCHAIN_API_KEY")
    return headers


def _get_endpoint() -> str:
    return os.getenv("LANGCHAIN_ENDPOINT", "http://localhost:8000")


class LangChainTracer(BaseTracer):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LangChain tracer."""
        super().__init__(**kwargs)
        self._endpoint = _get_endpoint()
        self._headers = _get_headers()

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        if isinstance(run, LLMRun):
            endpoint = f"{self._endpoint}/llm-runs"
        elif isinstance(run, ChainRun):
            endpoint = f"{self._endpoint}/chain-runs"
        else:
            endpoint = f"{self._endpoint}/tool-runs"

        try:
            response = requests.post(
                endpoint,
                data=run.json(),
                headers=self._headers,
            )
            raise_for_status_with_text(response)
        except Exception as e:
            logging.warning(f"Failed to persist run: {e}")

    def _persist_session(
        self, session_create: TracerSessionBase
    ) -> Union[TracerSession, TracerSessionV2]:
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

    def _load_session(self, session_name: Optional[str] = None) -> TracerSession:
        """Load a session from the tracer."""
        try:
            url = f"{self._endpoint}/sessions"
            if session_name:
                url += f"?name={session_name}"
            r = requests.get(url, headers=self._headers)

            tracer_session = TracerSession(**r.json()[0])
        except Exception as e:
            session_type = "default" if not session_name else session_name
            logging.warning(
                f"Failed to load {session_type} session, using empty session: {e}"
            )
            tracer_session = TracerSession(id=1)

        self.session = tracer_session
        return tracer_session

    def load_session(self, session_name: str) -> Union[TracerSession, TracerSessionV2]:
        """Load a session with the given name from the tracer."""
        return self._load_session(session_name)

    def load_default_session(self) -> Union[TracerSession, TracerSessionV2]:
        """Load the default tracing session and set it as the Tracer's session."""
        return self._load_session("default")


def _get_tenant_id() -> Optional[str]:
    """Get the tenant ID for the LangChain API."""
    tenant_id: Optional[str] = os.getenv("LANGCHAIN_TENANT_ID")
    if tenant_id:
        return tenant_id
    endpoint = _get_endpoint()
    headers = _get_headers()
    response = requests.get(endpoint + "/tenants", headers=headers)
    raise_for_status_with_text(response)
    tenants: List[Dict[str, Any]] = response.json()
    if not tenants:
        raise ValueError(f"No tenants found for URL {endpoint}")
    return tenants[0]["id"]


class LangChainTracerV2(LangChainTracer):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    def __init__(self, example_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """Initialize the LangChain tracer."""
        super().__init__(**kwargs)
        self._endpoint = _get_endpoint()
        self._headers = _get_headers()
        self.tenant_id = _get_tenant_id()
        self.example_id = example_id

    def _get_session_create(
        self, name: Optional[str] = None, **kwargs: Any
    ) -> TracerSessionBase:
        return TracerSessionV2Create(name=name, extra=kwargs, tenant_id=self.tenant_id)

    def _persist_session(self, session_create: TracerSessionBase) -> TracerSessionV2:
        """Persist a session."""
        session: Optional[TracerSessionV2] = None
        try:
            r = requests.post(
                f"{self._endpoint}/sessions",
                data=session_create.json(),
                headers=self._headers,
            )
            raise_for_status_with_text(r)
            creation_args = session_create.dict()
            if "id" in creation_args:
                del creation_args["id"]
            return TracerSessionV2(id=r.json()["id"], **creation_args)
        except Exception as e:
            if session_create.name is not None:
                try:
                    return self.load_session(session_create.name)
                except Exception:
                    pass
            logging.warning(
                f"Failed to create session {session_create.name},"
                f" using empty session: {e}"
            )
            session = TracerSessionV2(id=uuid4(), **session_create.dict())

        return session

    def _get_default_query_params(self) -> Dict[str, Any]:
        """Get the query params for the LangChain API."""
        return {"tenant_id": self.tenant_id}

    def load_session(self, session_name: str) -> TracerSessionV2:
        """Load a session with the given name from the tracer."""
        try:
            url = f"{self._endpoint}/sessions"
            params = {"tenant_id": self.tenant_id}
            if session_name:
                params["name"] = session_name
            r = requests.get(url, headers=self._headers, params=params)
            raise_for_status_with_text(r)
            tracer_session = TracerSessionV2(**r.json()[0])
        except Exception as e:
            session_type = "default" if not session_name else session_name
            logging.warning(
                f"Failed to load {session_type} session, using empty session: {e}"
            )
            tracer_session = TracerSessionV2(id=uuid4(), tenant_id=self.tenant_id)

        self.session = tracer_session
        return tracer_session

    def load_default_session(self) -> TracerSessionV2:
        """Load the default tracing session and set it as the Tracer's session."""
        return self.load_session("default")

    def _convert_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> RunCreate:
        """Convert a run to a Run."""
        session = self.session or self.load_default_session()
        inputs: Dict[str, Any] = {}
        outputs: Optional[Dict[str, Any]] = None
        child_runs: List[Union[LLMRun, ChainRun, ToolRun]] = []
        if isinstance(run, LLMRun):
            run_type = "llm"
            inputs = {"prompts": run.prompts}
            outputs = run.response.dict() if run.response else {}
            child_runs = []
        elif isinstance(run, ChainRun):
            run_type = "chain"
            inputs = run.inputs
            outputs = run.outputs
            child_runs = [
                *run.child_llm_runs,
                *run.child_chain_runs,
                *run.child_tool_runs,
            ]
        else:
            run_type = "tool"
            inputs = {"input": run.tool_input}
            outputs = {"output": run.output} if run.output else {}
            child_runs = [
                *run.child_llm_runs,
                *run.child_chain_runs,
                *run.child_tool_runs,
            ]

        return RunCreate(
            id=run.uuid,
            name=run.serialized.get("name"),
            start_time=run.start_time,
            end_time=run.end_time,
            extra=run.extra or {},
            error=run.error,
            execution_order=run.execution_order,
            serialized=run.serialized,
            inputs=inputs,
            outputs=outputs,
            session_id=session.id,
            run_type=run_type,
            child_runs=[self._convert_run(child) for child in child_runs],
        )

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        run_create = self._convert_run(run)
        run_create.reference_example_id = self.example_id
        try:
            response = requests.post(
                f"{self._endpoint}/runs",
                data=run_create.json(),
                headers=self._headers,
            )
            raise_for_status_with_text(response)
        except Exception as e:
            logging.warning(f"Failed to persist run: {e}")
