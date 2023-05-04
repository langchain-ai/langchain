from __future__ import annotations

import asyncio
import logging
import socket
from io import BytesIO
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional,
                    Sequence, Tuple, Union)
from urllib.parse import urlsplit

import aiohttp
import requests
from pydantic import BaseSettings, Field, root_validator
from requests import HTTPError, Response

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.tracers.langchain import LangChainTracerV2
from langchain.callbacks.tracers.schemas import (TracerSession,
                                                 TracerSessionCreate)
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel
from langchain.client.models import Dataset, Example
from langchain.client.utils import parse_chat_messages
from langchain.llms.base import BaseLLM
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
    api_url: str = Field(..., env="LANGCHAIN_ENDPOINT")
    tenant_id: str = Field(..., env="LANGCHAIN_TENANT_ID")

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
        else:
            tenant_id = values.get("tenant_id")
            if not tenant_id:
                values["tenant_id"] = cls._get_seeded_tenant_id(api_url, api_key)     
        return values
    
    @staticmethod
    def _get_seeded_tenant_id(api_url: str, api_key: Optional[str]) -> str:
        """Get the tenant ID from the seeded tenant."""
        url = f"{api_url}/tenants"
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(url, headers=headers)
        _raise_rich_error(response)
        results: List[dict] = response.json()
        if len(results) == 0:
            raise ValueError("No seeded tenant found")
        return results[0]["id"]

    def _repr_html_(self) -> str:
        """Return an HTML representation of the instance with a link to the URL."""
        link = _get_link_stem(self.api_url)
        return f'<a href="{link}", target="_blank" rel="noopener">LangChain+ Client</a>'

    def __repr__(self) -> str:
        """Return a string representation of the instance with a link to the URL."""
        return f"LangChainPlusClient (API URL: {self.api_url})"

    @property
    def _headers(self) -> Dict[str, str]:
        """Get the headers for the API request."""
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    @xor_args(("session_name", "session_id"))
    def read_session(
        self, *, session_name: Optional[str] = None, session_id: Optional[int] = None
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
                return self.read_session(session_name=session_name)
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
        """Upload a dataframe as a CSV to the LangChain+ API."""
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
        _raise_rich_error(response)
        return response.json()

    @xor_args(("dataset_name", "dataset_id"))
    def read_dataset(
        self, *, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> Dataset:
        url = f"{self.api_url}/datasets"
        params: Dict[str, Any] = {"limit": 1}
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

    @xor_args(("dataset_name", "dataset_id"))
    async def aread_dataset(
        self, *, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> Dataset:
        """Read a dataset from the LangChain+ API."""
        url = f"{self.api_url}/datasets"
        params: Dict[str, Any] = {"limit": 1}
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

    # Examples APIs.

    def read_example(self, example_id: str) -> Example:
        """Read an example from the LangChain+ API."""
        response = requests.get(
            self.api_url + f"/examples/{example_id}", headers=self._headers
        )
        _raise_rich_error(response)
        return Example(**response.json())

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
            
    @staticmethod
    async def _arun_llm(llm: BaseLanguageModel, inputs: Dict[str, Any], langchain_tracer: LangChainTracerV2,)-> Union[Dict, str]:
        if isinstance(llm, BaseLLM):
            llm_prompts: List[str] = inputs['prompts']
            chain_output = await llm.agenerate(llm_prompts, callbacks=[langchain_tracer])
        elif isinstance(llm, BaseChatModel):
            chat_prompts: List[str] = inputs['prompts']
            messages = [parse_chat_messages(chat_prompt) for chat_prompt in chat_prompts]
            chain_output = await llm.agenerate(messages, callbacks=[langchain_tracer])
        else:
            raise ValueError(f"Unsupported LLM type {type(llm)}")
        return chain_output

    @staticmethod
    async def _arun_llm_or_chain(
        example: Example, langchain_tracer: LangChainTracerV2, llm_or_chain: Union[Chain, BaseLanguageModel], n_times: int
    ) -> List[Union[dict, str]]:
        """Run the chain asynchronously."""
        previous_example_id = langchain_tracer.example_id
        langchain_tracer.example_id = example.id
        outputs = []
        for  _ in range(n_times):
            try:
                if isinstance(llm_or_chain, BaseLanguageModel):
                    chain_output = await LangChainPlusClient._arun_llm(llm_or_chain, example.inputs)
                else:
                    chain_output = await llm_or_chain.arun(
                        example.inputs, callbacks=[langchain_tracer]
                    )
                outputs.append(chain_output)
            except Exception as e:
                logger.warning(f"Chain failed for example {example.id}. Error: {e}")
                outputs.append({"Error": str(e)})
            finally:
                langchain_tracer.example_id = previous_example_id
        return outputs 

    @staticmethod
    async def _worker(
        queue: asyncio.Queue,
        tracers: List[LangChainTracerV2],
        chain: Chain,
        results: Dict[str, Any],
    ) -> None:
        """Worker for running the chain on examples."""
        while True:
            example: Optional[Example] = await queue.get()
            if example is None:
                break

            tracer = tracers.pop()
            result = await LangChainPlusClient._arun_llm_or_chain(example, tracer, chain)
            results[example.id] = result
            tracers.append(tracer)
            queue.task_done()

    @staticmethod
    async def _arun_llm_or_chain_over_buffer(
        buffer: Sequence[Example], tracers: Sequence[LangChainTracerV2], llm_or_chain: Union[Chain, BaseLanguageModel],
    ) -> List[Union[str, dict]]:
        """Run the chain asynchronously over a buffer of examples."""
        batch_results = [
            LangChainPlusClient._arun_llm_or_chain(example, tracer, llm_or_chain)
            for example, tracer in zip(buffer, tracers)
        ]
        return await asyncio.gather(*batch_results)

    async def arun_on_dataset(
        self,
        dataset_name: str,
        llm_or_chain: Union[Chain, BaseLanguageModel],
        num_workers: int = 5,
        num_repetitions: int = 1,
        session_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the chain on a dataset and store traces to the specified session name.

        Args:
            dataset_name: Name of the dataset to run the chain on.
            llm_or_chain: Chain or language model to run over the dataset.
            num_workers: Number of async workers to run in parallel.
            num_repetitions: Number of times to run the chain on each example.
            session_name: Name of the session to store the traces in.

        Returns:
            A dictionary mapping example ids to the chain outputs.
        """
        if isinstance(llm_or_chain, BaseLanguageModel):
            raise NotImplementedError
        if session_name is None:
            session_name = f"{dataset_name}_{llm_or_chain.__class__.__name__}-{num_repetitions}"
        self.create_session(session_name)

        dataset = await self.aread_dataset(dataset_name=dataset_name)

        tracers = []
        for _ in range(num_workers):
            tracer = LangChainTracerV2()
            tracer.load_session(session_name or "default")
            tracers.append(tracer)

        results: Dict[str, Any] = {}
        examples = await self.alist_examples(dataset_id=dataset.id)

        queue: asyncio.Queue[Optional[Example]] = asyncio.Queue()
        workers = [
            asyncio.create_task(LangChainPlusClient._worker(queue, tracers, llm_or_chain, results))
            for _ in range(num_workers)
        ]

        for example in examples:
            await queue.put(example)

        await queue.join()  # Wait for all tasks to complete

        for _ in workers:
            await queue.put(None)  # Signal the workers to exit

        await asyncio.gather(*workers)

        return results
