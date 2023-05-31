"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import requests
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import (
    Run,
    RunCreate,
    RunTypeEnum,
    RunUpdate,
    TracerSession,
    TracerSessionCreate,
)
from langchain.schema import BaseMessage, messages_to_dict
from langchain.utils import raise_for_status_with_text

logger = logging.getLogger(__name__)


def get_headers() -> Dict[str, Any]:
    """Get the headers for the LangChain API."""
    headers: Dict[str, Any] = {"Content-Type": "application/json"}
    if os.getenv("LANGCHAIN_API_KEY"):
        headers["x-api-key"] = os.getenv("LANGCHAIN_API_KEY")
    return headers


def get_endpoint() -> str:
    return os.getenv("LANGCHAIN_ENDPOINT", "http://localhost:1984")


class LangChainTracerAPIError(Exception):
    """An error occurred while communicating with the LangChain API."""


class LangChainTracerUserError(Exception):
    """An error occurred while communicating with the LangChain API."""


class LangChainTracerError(Exception):
    """An error occurred while communicating with the LangChain API."""


retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(LangChainTracerAPIError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


@retry_decorator
def _get_tenant_id(
    tenant_id: Optional[str], endpoint: Optional[str], headers: Optional[dict]
) -> str:
    """Get the tenant ID for the LangChain API."""
    tenant_id_: Optional[str] = tenant_id or os.getenv("LANGCHAIN_TENANT_ID")
    if tenant_id_:
        return tenant_id_
    endpoint_ = endpoint or get_endpoint()
    headers_ = headers or get_headers()
    response = None
    try:
        response = requests.get(endpoint_ + "/tenants", headers=headers_)
        raise_for_status_with_text(response)
    except HTTPError as e:
        if response is not None and response.status_code == 500:
            raise LangChainTracerAPIError(
                f"Failed to get tenant ID from LangChain API. {e}"
            )
        else:
            raise LangChainTracerUserError(
                f"Failed to get tenant ID from LangChain API. {e}"
            )
    except Exception as e:
        raise LangChainTracerError(
            f"Failed to get tenant ID from LangChain API. {e}"
        ) from e

    tenants: List[Dict[str, Any]] = response.json()
    if not tenants:
        raise ValueError(f"No tenants found for URL {endpoint_}")
    return tenants[0]["id"]


class LangChainTracer(BaseTracer):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        example_id: Optional[UUID] = None,
        session_name: Optional[str] = None,
        session_extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangChain tracer."""
        super().__init__(**kwargs)
        self.session: Optional[TracerSession] = None
        self._endpoint = get_endpoint()
        self._headers = get_headers()
        self.tenant_id = tenant_id
        self.example_id = example_id
        self.session_name = session_name or os.getenv("LANGCHAIN_SESSION", "default")
        self.session_extra = session_extra
        # set max_workers to 1 to process tasks in order
        self.executor = ThreadPoolExecutor(max_workers=1)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for an LLM run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        chat_model_run = Run(
            id=run_id,
            name=serialized.get("name"),
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [messages_to_dict(batch) for batch in messages]},
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=execution_order,
            child_execution_order=execution_order,
            run_type=RunTypeEnum.llm,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)

    def ensure_tenant_id(self) -> str:
        """Load or use the tenant ID."""
        tenant_id = self.tenant_id or _get_tenant_id(
            self.tenant_id, self._endpoint, self._headers
        )
        self.tenant_id = tenant_id
        return tenant_id

    @retry_decorator
    def ensure_session(self) -> TracerSession:
        """Upsert a session."""
        if self.session is not None:
            return self.session
        tenant_id = self.ensure_tenant_id()
        url = f"{self._endpoint}/sessions?upsert=true"
        session_create = TracerSessionCreate(
            name=self.session_name, extra=self.session_extra, tenant_id=tenant_id
        )
        response = None
        try:
            response = requests.post(
                url,
                data=session_create.json(),
                headers=self._headers,
            )
            response.raise_for_status()
        except HTTPError as e:
            if response is not None and response.status_code == 500:
                raise LangChainTracerAPIError(
                    f"Failed to upsert session to LangChain API. {e}"
                )
            else:
                raise LangChainTracerUserError(
                    f"Failed to upsert session to LangChain API. {e}"
                )
        except Exception as e:
            raise LangChainTracerError(
                f"Failed to upsert session to LangChain API. {e}"
            ) from e
        self.session = TracerSession(**response.json())
        return self.session

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""

    @retry_decorator
    def _persist_run_single(self, run: Run) -> None:
        """Persist a run."""
        session = self.ensure_session()
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        run_dict = run.dict()
        del run_dict["child_runs"]
        run_create = RunCreate(**run_dict, session_id=session.id)
        response = None
        try:
            response = requests.post(
                f"{self._endpoint}/runs",
                data=run_create.json(),
                headers=self._headers,
            )
            response.raise_for_status()
        except HTTPError as e:
            if response is not None and response.status_code == 500:
                raise LangChainTracerAPIError(
                    f"Failed to upsert persist run to LangChain API. {e}"
                )
            else:
                raise LangChainTracerUserError(
                    f"Failed to persist run to LangChain API. {e}"
                )
        except Exception as e:
            raise LangChainTracerError(
                f"Failed to persist run to LangChain API. {e}"
            ) from e

    @retry_decorator
    def _update_run_single(self, run: Run) -> None:
        """Update a run."""
        run_update = RunUpdate(**run.dict())
        response = None
        try:
            response = requests.patch(
                f"{self._endpoint}/runs/{run.id}",
                data=run_update.json(),
                headers=self._headers,
            )
            response.raise_for_status()
        except HTTPError as e:
            if response is not None and response.status_code == 500:
                raise LangChainTracerAPIError(
                    f"Failed to update run to LangChain API. {e}"
                )
            else:
                raise LangChainTracerUserError(f"Failed to run to LangChain API. {e}")
        except Exception as e:
            raise LangChainTracerError(
                f"Failed to update run to LangChain API. {e}"
            ) from e

    def _on_llm_start(self, run: Run) -> None:
        """Persist an LLM run."""
        self.executor.submit(self._persist_run_single, run.copy(deep=True))

    def _on_chat_model_start(self, run: Run) -> None:
        """Persist an LLM run."""
        self.executor.submit(self._persist_run_single, run.copy(deep=True))

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""
        self.executor.submit(self._update_run_single, run.copy(deep=True))

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""
        self.executor.submit(self._update_run_single, run.copy(deep=True))

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""
        self.executor.submit(self._persist_run_single, run.copy(deep=True))

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""
        self.executor.submit(self._update_run_single, run.copy(deep=True))

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""
        self.executor.submit(self._update_run_single, run.copy(deep=True))

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""
        self.executor.submit(self._persist_run_single, run.copy(deep=True))

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""
        self.executor.submit(self._update_run_single, run.copy(deep=True))

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""
        self.executor.submit(self._update_run_single, run.copy(deep=True))
