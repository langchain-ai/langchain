from __future__ import annotations

import logging
import socket
from datetime import datetime
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlsplit
from uuid import UUID

import requests
from pydantic import BaseSettings, Field, root_validator
from requests import Response
from tenacity import retry, stop_after_attempt, wait_fixed

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.tracers.schemas import Run, TracerSession
from langchain.chains.base import Chain
from langchain.client.models import (
    Dataset,
    DatasetCreate,
    Example,
    ExampleCreate,
    ListRunsQueryParams,
)
from langchain.client.runner_utils import arun_on_examples, run_on_examples
from langchain.utils import raise_for_status_with_text, xor_args

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

MODEL_OR_CHAIN_FACTORY = Union[Callable[[], Chain], BaseLanguageModel]


def _get_link_stem(url: str) -> str:
    scheme = urlsplit(url).scheme
    netloc_prefix = urlsplit(url).netloc.split(":")[0]
    return f"{scheme}://{netloc_prefix}"


def _is_localhost(url: str) -> bool:
    """Check if the URL is localhost."""
    try:
        netloc = urlsplit(url).netloc.split(":")[0]
        ip = socket.gethostbyname(netloc)
        return ip == "127.0.0.1" or ip.startswith("0.0.0.0") or ip.startswith("::")
    except socket.gaierror:
        return False


class LangChainPlusClient(BaseSettings):
    """Client for interacting with the LangChain+ API."""

    api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    api_url: str = Field(default="http://localhost:8000", env="LANGCHAIN_ENDPOINT")
    tenant_id: Optional[str] = None

    @root_validator(pre=True)
    def validate_api_key_if_hosted(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify API key is provided if url not localhost."""
        api_url: str = values.get("api_url", "http://localhost:8000")
        api_key: Optional[str] = values.get("api_key")
        if not _is_localhost(api_url):
            if not api_key:
                raise ValueError(
                    "API key must be provided when using hosted LangChain+ API"
                )
        tenant_id = values.get("tenant_id")
        if not tenant_id:
            values["tenant_id"] = LangChainPlusClient._get_seeded_tenant_id(
                api_url, api_key
            )
        return values

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def _get_seeded_tenant_id(api_url: str, api_key: Optional[str]) -> str:
        """Get the tenant ID from the seeded tenant."""
        url = f"{api_url}/tenants"
        headers = {"x-api-key": api_key} if api_key else {}
        response = requests.get(url, headers=headers)
        try:
            raise_for_status_with_text(response)
        except Exception as e:
            raise ValueError(
                "Unable to get default tenant ID. Please manually provide."
            ) from e
        results: List[dict] = response.json()
        if len(results) == 0:
            raise ValueError("No seeded tenant found")
        return results[0]["id"]

    @staticmethod
    def _get_session_name(
        session_name: Optional[str],
        llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
        dataset_name: str,
    ) -> str:
        if session_name is not None:
            return session_name
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            model_name = llm_or_chain_factory.__class__.__name__
        else:
            model_name = llm_or_chain_factory().__class__.__name__
        return f"{dataset_name}-{model_name}-{current_time}"

    def _repr_html_(self) -> str:
        """Return an HTML representation of the instance with a link to the URL."""
        if _is_localhost(self.api_url):
            link = "http://localhost"
        elif "dev" in self.api_url:
            link = "https://dev.langchain.plus"
        else:
            link = "https://www.langchain.plus"
        return f'<a href="{link}", target="_blank" rel="noopener">LangChain+ Client</a>'

    def __repr__(self) -> str:
        """Return a string representation of the instance with a link to the URL."""
        return f"LangChainPlusClient (API URL: {self.api_url})"

    @property
    def _headers(self) -> Dict[str, str]:
        """Get the headers for the API request."""
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    @property
    def query_params(self) -> Dict[str, Any]:
        """Get the headers for the API request."""
        return {"tenant_id": self.tenant_id}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Response:
        """Make a GET request."""
        query_params = self.query_params
        if params:
            query_params.update(params)
        return requests.get(
            f"{self.api_url}{path}", headers=self._headers, params=query_params
        )

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        input_keys: List[str],
        output_keys: List[str],
    ) -> Dataset:
        """Upload a dataframe as individual examples to the LangChain+ API."""
        dataset = self.create_dataset(dataset_name=name, description=description)
        for row in df.itertuples():
            inputs = {key: getattr(row, key) for key in input_keys}
            outputs = {key: getattr(row, key) for key in output_keys}
            self.create_example(inputs, outputs=outputs, dataset_id=dataset.id)
        return dataset

    def upload_csv(
        self,
        csv_file: Union[str, Tuple[str, BytesIO]],
        description: str,
        input_keys: List[str],
        output_keys: List[str],
    ) -> Dataset:
        """Upload a CSV file to the LangChain+ API."""
        files = {"file": csv_file}
        data = {
            "input_keys": ",".join(input_keys),
            "output_keys": ",".join(output_keys),
            "description": description,
            "tenant_id": self.tenant_id,
        }
        response = requests.post(
            self.api_url + "/datasets/upload",
            headers=self._headers,
            data=data,
            files=files,
        )
        raise_for_status_with_text(response)
        result = response.json()
        # TODO: Make this more robust server-side
        if "detail" in result and "already exists" in result["detail"]:
            file_name = csv_file if isinstance(csv_file, str) else csv_file[0]
            file_name = file_name.split("/")[-1]
            raise ValueError(f"Dataset {file_name} already exists")
        return Dataset(**result)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def read_run(self, run_id: str) -> Run:
        """Read a run from the LangChain+ API."""
        response = self._get(f"/runs/{run_id}")
        raise_for_status_with_text(response)
        return Run(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def list_runs(
        self,
        *,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        run_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[Run]:
        """List runs from the LangChain+ API."""
        if session_name is not None:
            if session_id is not None:
                raise ValueError("Only one of session_id or session_name may be given")
            session_id = self.read_session(session_name=session_name).id
        query_params = ListRunsQueryParams(
            session_id=session_id, run_type=run_type, **kwargs
        )
        filtered_params = {
            k: v for k, v in query_params.dict().items() if v is not None
        }
        response = self._get("/runs", params=filtered_params)
        raise_for_status_with_text(response)
        yield from [Run(**run) for run in response.json()]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    @xor_args(("session_id", "session_name"))
    def read_session(
        self, *, session_id: Optional[str] = None, session_name: Optional[str] = None
    ) -> TracerSession:
        """Read a session from the LangChain+ API."""
        path = "/sessions"
        params: Dict[str, Any] = {"limit": 1, "tenant_id": self.tenant_id}
        if session_id is not None:
            path += f"/{session_id}"
        elif session_name is not None:
            params["name"] = session_name
        else:
            raise ValueError("Must provide dataset_name or dataset_id")
        response = self._get(
            path,
            params=params,
        )
        raise_for_status_with_text(response)
        response = self._get(
            path,
            params=params,
        )
        raise_for_status_with_text(response)
        result = response.json()
        if isinstance(result, list):
            if len(result) == 0:
                raise ValueError(f"Dataset {session_name} not found")
            return TracerSession(**result[0])
        return TracerSession(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def list_sessions(self) -> Iterator[TracerSession]:
        """List sessions from the LangChain+ API."""
        response = self._get("/sessions")
        raise_for_status_with_text(response)
        yield from [TracerSession(**session) for session in response.json()]

    def create_dataset(self, dataset_name: str, description: str) -> Dataset:
        """Create a dataset in the LangChain+ API."""
        dataset = DatasetCreate(
            tenant_id=self.tenant_id,
            name=dataset_name,
            description=description,
        )
        response = requests.post(
            self.api_url + "/datasets",
            headers=self._headers,
            data=dataset.json(),
        )
        raise_for_status_with_text(response)
        return Dataset(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    @xor_args(("dataset_name", "dataset_id"))
    def read_dataset(
        self, *, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> Dataset:
        path = "/datasets"
        params: Dict[str, Any] = {"limit": 1, "tenant_id": self.tenant_id}
        if dataset_id is not None:
            path += f"/{dataset_id}"
        elif dataset_name is not None:
            params["name"] = dataset_name
        else:
            raise ValueError("Must provide dataset_name or dataset_id")
        response = self._get(
            path,
            params=params,
        )
        raise_for_status_with_text(response)
        result = response.json()
        if isinstance(result, list):
            if len(result) == 0:
                raise ValueError(f"Dataset {dataset_name} not found")
            return Dataset(**result[0])
        return Dataset(**result)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def list_datasets(self, limit: int = 100) -> Iterator[Dataset]:
        """List the datasets on the LangChain+ API."""
        response = self._get("/datasets", params={"limit": limit})
        raise_for_status_with_text(response)
        yield from [Dataset(**dataset) for dataset in response.json()]

    @xor_args(("dataset_id", "dataset_name"))
    def delete_dataset(
        self, *, dataset_id: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> Dataset:
        """Delete a dataset by ID or name."""
        if dataset_name is not None:
            dataset_id = self.read_dataset(dataset_name=dataset_name).id
        if dataset_id is None:
            raise ValueError("Must provide either dataset name or ID")
        response = requests.delete(
            f"{self.api_url}/datasets/{dataset_id}",
            headers=self._headers,
        )
        raise_for_status_with_text(response)
        return Dataset(**response.json())

    @xor_args(("dataset_id", "dataset_name"))
    def create_example(
        self,
        inputs: Dict[str, Any],
        dataset_id: Optional[UUID] = None,
        dataset_name: Optional[str] = None,
        created_at: Optional[datetime] = None,
        outputs: Dict[str, Any] | None = None,
    ) -> Example:
        """Create a dataset example in the LangChain+ API."""
        if dataset_id is None:
            dataset_id = self.read_dataset(dataset_name).id

        data = {
            "inputs": inputs,
            "outputs": outputs,
            "dataset_id": dataset_id,
        }
        if created_at:
            data["created_at"] = created_at.isoformat()
        example = ExampleCreate(**data)
        response = requests.post(
            f"{self.api_url}/examples", headers=self._headers, data=example.json()
        )
        raise_for_status_with_text(response)
        result = response.json()
        return Example(**result)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def read_example(self, example_id: str) -> Example:
        """Read an example from the LangChain+ API."""
        response = self._get(f"/examples/{example_id}")
        raise_for_status_with_text(response)
        return Example(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def list_examples(
        self, dataset_id: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> Iterator[Example]:
        """List the datasets on the LangChain+ API."""
        params = {}
        if dataset_id is not None:
            params["dataset"] = dataset_id
        elif dataset_name is not None:
            dataset_id = self.read_dataset(dataset_name=dataset_name).id
            params["dataset"] = dataset_id
        else:
            pass
        response = self._get("/examples", params=params)
        raise_for_status_with_text(response)
        yield from [Example(**dataset) for dataset in response.json()]

    async def arun_on_dataset(
        self,
        dataset_name: str,
        llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
        *,
        concurrency_level: int = 5,
        num_repetitions: int = 1,
        session_name: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the chain on a dataset and store traces to the specified session name.

        Args:
            dataset_name: Name of the dataset to run the chain on.
            llm_or_chain_factory: Language model or Chain constructor to run
                over the dataset. The Chain constructor is used to permit
                independent calls on each example without carrying over state.
            concurrency_level: The number of async tasks to run concurrently.
            num_repetitions: Number of times to run the model on each example.
                This is useful when testing success rates or generating confidence
                intervals.
            session_name: Name of the session to store the traces in.
                Defaults to {dataset_name}-{chain class name}-{datetime}.
            verbose: Whether to print progress.

        Returns:
            A dictionary mapping example ids to the model outputs.
        """
        session_name = LangChainPlusClient._get_session_name(
            session_name, llm_or_chain_factory, dataset_name
        )
        dataset = self.read_dataset(dataset_name=dataset_name)
        examples = self.list_examples(dataset_id=str(dataset.id))

        return await arun_on_examples(
            examples,
            llm_or_chain_factory,
            concurrency_level=concurrency_level,
            num_repetitions=num_repetitions,
            session_name=session_name,
            verbose=verbose,
        )

    def run_on_dataset(
        self,
        dataset_name: str,
        llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
        *,
        num_repetitions: int = 1,
        session_name: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run the chain on a dataset and store traces to the specified session name.

        Args:
            dataset_name: Name of the dataset to run the chain on.
            llm_or_chain_factory: Language model or Chain constructor to run
                over the dataset. The Chain constructor is used to permit
                independent calls on each example without carrying over state.
            concurrency_level: Number of async workers to run in parallel.
            num_repetitions: Number of times to run the model on each example.
                This is useful when testing success rates or generating confidence
                intervals.
            session_name: Name of the session to store the traces in.
                Defaults to {dataset_name}-{chain class name}-{datetime}.
            verbose: Whether to print progress.

        Returns:
            A dictionary mapping example ids to the model outputs.
        """
        session_name = LangChainPlusClient._get_session_name(
            session_name, llm_or_chain_factory, dataset_name
        )
        dataset = self.read_dataset(dataset_name=dataset_name)
        examples = self.list_examples(dataset_id=str(dataset.id))
        return run_on_examples(
            examples,
            llm_or_chain_factory,
            num_repetitions=num_repetitions,
            session_name=session_name,
            verbose=verbose,
        )
