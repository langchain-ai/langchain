from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import aiohttp
import requests
from pydantic import BaseSettings, Field, root_validator
from requests import HTTPError, Response

from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.schemas import TracerSession, TracerSessionCreate
from langchain.chains.base import Chain
from langchain.client.models import Dataset, Example
from langchain.utils import xor_args

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _raise_rich_error(response: Response) -> None:
    """Raise an error with the response text."""
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e


class LangChainPlusClient(BaseSettings):
    """Client for interacting with the LangChain+ API."""

    api_key: str = Field(default=None, env="LANGCHAIN_API_KEY")
    api_url: str = Field(default="http://localhost:8000", env="LANGCHAIN_ENDPOINT")

    @root_validator
    def validate_api_key_if_hosted(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["api_url"] != "http://localhost:8000" and not values["api_key"]:
            raise ValueError(
                "API key must be provided when using hosted LangChain+ API"
            )
        return values

    @property
    def _headers(self) -> Dict[str, str]:
        """Get the headers for the API request."""
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    async def _arun_chain(
        example: Example, langchain_tracer: LangChainTracer, chain: Chain
    ) -> dict:
        """Run the chain asynchronously"""
        previous_example_id = langchain_tracer.example_id
        langchain_tracer.example_id = example.id
        try:
            chain_output = await chain.arun(
                example.inputs, callbacks=[langchain_tracer]
            )
        except Exception as e:
            logger.warning(f"Chain failed for example {example.id}. Error: {e}")
            return {"Error": str(e)}
        finally:
            langchain_tracer.example_id = previous_example_id
        return chain_output

    @staticmethod
    async def _arun_chain_over_buffer(
        buffer: Sequence[Example], tracers: Sequence[LangChainTracer], chain: Chain
    ) -> List[Any]:
        """Run the chain asynchronously over a buffer of examples."""
        batch_results = [
            LangChainPlusClient._arun_chain(example, tracer, chain)
            for example, tracer in zip(buffer, tracers)
        ]
        return await asyncio.gather(*batch_results)

    @xor_args(["session_name", "session_id"])
    def get_session(
        self, *, session_name: str = None, session_id: int = None
    ) -> TracerSession:
        """Get a session by name."""
        url = f"{self.api_url}/sessions"
        params = {}
        if session_id is not None:
            url = f"{url}/{session_id}"
        elif session_name is not None:
            params["name"] = session_name
        else:
            raise ValueError("Must provide either session name or ID")
        response = requests.get(url, headers=self._headers, params=params)
        _raise_rich_error(response)
        results = response.json()
        if isinstance(results, list):
            if len(results) == 0:
                raise ValueError(f"No session found with name {session_name}")
            return TracerSession(**results[0])
        return TracerSession(**results)

    def create_session(self, session_name: str) -> TracerSession:
        """Persist a session."""
        session_create = TracerSessionCreate(name=session_name)
        try:
            response = requests.post(
                f"{self.api_url}/sessions",
                data=session_create.json(),
                headers=self._headers,
            )
            result = response.json()
            if "id" in result:
                session = TracerSession(id=result["id"], **session_create.dict())
            elif "detail" in result and "already exists" in result["detail"]:
                return self.get_session(session_name=session_name)
            else:
                raise ValueError(f"Failed to create session: {result}")
        except Exception as e:
            logging.warning(
                f"Failed to create session '{session_name}', using default session: {e}"
            )
            session = TracerSession(id=1, **session_create.dict())
        return session

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        input_keys: List[str],
        output_keys: List[str],
    ) -> Dataset:
        if not name.endswith(".csv"):
            raise ValueError("Name must end with .csv")
        if not all([key in df.columns for key in input_keys]):
            raise ValueError("Input keys must be in dataframe columns")
        if not all([key in df.columns for key in output_keys]):
            raise ValueError("Output keys must be in dataframe columns")
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        csv_file = (name, buffer)
        return self.upload_csv(
            csv_file, description, input_keys=input_keys, output_keys=output_keys
        )

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
        }
        response = requests.post(
            self.api_url + "/datasets/upload",
            headers=self._headers,
            data=data,
            files=files,
        )
        _raise_rich_error(response)
        result = response.json()
        # TODO: Make this more robust server-side
        if "detail" in result and "already exists" in result["detail"]:
            file_name = csv_file if isinstance(csv_file, str) else csv_file[0]
            file_name = file_name.split("/")[-1]
            raise ValueError(f"Dataset {file_name} already exists")
        return Dataset(**result)

    def list_datasets(self, limit: int = 100) -> Iterable[Dataset]:
        """List the datasets on the LangChain+ API."""
        response = requests.get(
            self.api_url + "/datasets", headers=self._headers, params={"limit": limit}
        )
        _raise_rich_error(response)
        return [Dataset(**dataset) for dataset in response.json()]

    async def alist_datasets(self, limit: int = 100) -> Iterable[Dataset]:
        """List the datasets on the LangChain+ API."""
        async with aiohttp.ClientSession() as session:
            async with session.request(
                "get",
                self.api_url + "/datasets",
                headers=self._headers,
                params={"limit": limit},
            ) as response:
                response.raise_for_status()
                results = await response.json()
                return [Dataset(**dataset) for dataset in results]

    def list_examples(self, dataset_id: Optional[str] = None) -> Iterable[Example]:
        """List the datasets on the LangChain+ API."""
        params = {} if dataset_id is None else {"dataset": dataset_id}
        response = requests.get(
            self.api_url + "/examples", headers=self._headers, params=params
        )
        _raise_rich_error(response)
        return [Example(**dataset) for dataset in response.json()]

    async def alist_examples(
        self, dataset_id: Optional[str] = None
    ) -> Iterable[Example]:
        """List the datasets on the LangChain+ API."""
        params = {} if dataset_id is None else {"dataset": dataset_id}
        async with aiohttp.ClientSession() as session:
            async with session.request(
                "get", self.api_url + "/examples", headers=self._headers, params=params
            ) as response:
                response.raise_for_status()
                results = await response.json()
                return [Example(**dataset) for dataset in results]

    @xor_args(["dataset_name", "dataset_id"])
    def read_dataset(
        self, *, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> Dataset:
        url = f"{self.api_url}/datasets"
        params = {"limit": 1}
        if dataset_id is not None:
            url += f"/{dataset_id}"
        elif dataset_name is not None:
            params["name"] = dataset_name
        else:
            raise ValueError("Must provide dataset_name or dataset_id")
        response = requests.get(
            url,
            params=params,
            headers=self._headers,
        )
        _raise_rich_error(response)
        result = response.json()
        if isinstance(result, list):
            if len(result) == 0:
                raise ValueError(f"Dataset {dataset_name} not found")
            return Dataset(**result[0])
        return Dataset(**result)

    @xor_args(["dataset_name", "dataset_id"])
    async def aread_dataset(
        self, *, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> Dataset:
        url = f"{self.api_url}/datasets"
        params = {"limit": 1}
        if dataset_id is not None:
            url += f"/{dataset_id}"
        elif dataset_name is not None:
            params["name"] = dataset_name
        async with aiohttp.ClientSession() as session:
            async with session.request(
                "get",
                url,
                params=params,
                headers=self._headers,
            ) as response:
                result = await response.json()
                if isinstance(result, list):
                    if len(result) == 0:
                        raise ValueError(f"Dataset {dataset_name} not found")
                    return Dataset(**result[0])
                return Dataset(**result)

    async def arun_chain_on_dataset(
        self,
        dataset_name: str,
        chain: Chain,
        batch_size: int = 5,
        session_name: Optional[str] = None,
    ) -> List[Any]:
        """Run the chain on the specified dataset"""
        if session_name is not None:
            self.create_session(session_name)
        dataset = await self.aread_dataset(dataset_name=dataset_name)
        tracers = [LangChainTracer() for _ in range(batch_size)]
        for tracer in tracers:
            tracer.load_session(session_name or "default")
        graded_outputs = []
        examples = await self.alist_examples(dataset_id=dataset.id)
        buffer = []
        for example in examples:
            buffer.append(example)
            if len(buffer) == batch_size:
                batch_results = await self._arun_chain_over_buffer(
                    buffer, tracers, chain
                )
                graded_outputs.extend(batch_results)
                buffer = []
        if buffer:
            batch_results = await self._arun_chain_over_buffer(buffer, tracers, chain)
            graded_outputs.extend(batch_results)
        return graded_outputs
