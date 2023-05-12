"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import requests

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import (
    Run,
    RunCreate,
    TracerSession,
    TracerSessionCreate,
    TracerSessionV1Base,
)
from langchain.utils import raise_for_status_with_text


def get_headers() -> Dict[str, Any]:
    """Get the headers for the LangChain API."""
    headers: Dict[str, Any] = {"Content-Type": "application/json"}
    if os.getenv("LANGCHAIN_API_KEY"):
        headers["x-api-key"] = os.getenv("LANGCHAIN_API_KEY")
    return headers


def get_endpoint() -> str:
    return os.getenv("LANGCHAIN_ENDPOINT", "http://localhost:8000")


def _get_tenant_id() -> Optional[str]:
    """Get the tenant ID for the LangChain API."""
    tenant_id: Optional[str] = os.getenv("LANGCHAIN_TENANT_ID")
    if tenant_id:
        return tenant_id
    endpoint = get_endpoint()
    headers = get_headers()
    response = requests.get(endpoint + "/tenants", headers=headers)
    raise_for_status_with_text(response)
    tenants: List[Dict[str, Any]] = response.json()
    if not tenants:
        raise ValueError(f"No tenants found for URL {endpoint}")
    return tenants[0]["id"]


class LangChainTracer(BaseTracer):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        example_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangChain tracer."""
        super().__init__(**kwargs)
        self.session: Optional[TracerSession] = None
        self._endpoint = get_endpoint()
        self._headers = get_headers()
        self.tenant_id = tenant_id or _get_tenant_id()
        self.example_id = example_id

    def new_session(self, name: Optional[str] = None, **kwargs: Any) -> TracerSession:
        """NOT thread safe, do not call this method from multiple threads."""
        session_create = TracerSessionCreate(
            name=name, extra=kwargs, tenant_id=self.tenant_id
        )
        session = self._persist_session(session_create)
        self.session = session
        return session

    def _persist_session(self, session_create: TracerSessionV1Base) -> TracerSession:
        """Persist a session."""
        session: Optional[TracerSession] = None
        try:
            r = requests.post(
                f"{self._endpoint}/sessions?upsert=true",
                data=session_create.json(),
                headers=self._headers,
            )
            raise_for_status_with_text(r)
            return TracerSession(**{**session_create.dict(), "id": r.json()["id"]})
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
            session = TracerSession(**{**session_create.dict(), "id": uuid4()})

        return session

    def _get_default_query_params(self) -> Dict[str, Any]:
        """Get the query params for the LangChain API."""
        return {"tenant_id": self.tenant_id}

    def load_session(self, session_name: str) -> TracerSession:
        """Load a session with the given name from the tracer."""
        try:
            url = f"{self._endpoint}/sessions"
            params = {"tenant_id": self.tenant_id}
            if session_name:
                params["name"] = session_name
            r = requests.get(url, headers=self._headers, params=params)
            raise_for_status_with_text(r)
            tracer_session = TracerSession(**r.json()[0])
        except Exception as e:
            session_type = "default" if not session_name else session_name
            logging.warning(
                f"Failed to load {session_type} session, using empty session: {e}"
            )
            tracer_session = TracerSession(id=uuid4(), tenant_id=self.tenant_id)

        self.session = tracer_session
        return tracer_session

    def load_default_session(self) -> TracerSession:
        """Load the default tracing session and set it as the Tracer's session."""
        return self.load_session("default")

    def _persist_run_nested(self, run: Run) -> None:
        """Persist a run."""
        if self.session is None:
            self.session = self.load_default_session()
        child_runs = run.child_runs
        run_dict = run.dict()
        del run_dict["child_runs"]
        run_create = RunCreate(**run_dict, session_id=self.session.id)
        try:
            response = requests.post(
                f"{self._endpoint}/runs",
                data=run_create.json(),
                headers=self._headers,
            )
            raise_for_status_with_text(response)
        except Exception as e:
            logging.warning(f"Failed to persist run: {e}")
        for child_run in child_runs:
            child_run.parent_run_id = run.id
            self._persist_run_nested(child_run)

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        run.reference_example_id = self.example_id
        # TODO: Post first then patch
        self._persist_run_nested(run)
