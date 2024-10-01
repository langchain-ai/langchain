"""Unstructured document loader."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import IO, Any, Callable, Iterator, Optional, cast

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from typing_extensions import TypeAlias
from unstructured_client import UnstructuredClient  # type: ignore
from unstructured_client.models import operations, shared  # type: ignore

Element: TypeAlias = Any

logger = logging.getLogger(__file__)

_DEFAULT_URL = "https://api.unstructuredapp.io/general/v0/general"


class UnstructuredLoader(BaseLoader):
    """Unstructured document loader interface.

    Setup:
        Install ``langchain-unstructured`` and set environment variable ``UNSTRUCTURED_API_KEY``.

        .. code-block:: bash
            pip install -U langchain-unstructured
            export UNSTRUCTURED_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python
            from langchain_unstructured import UnstructuredLoader

            loader = UnstructuredLoader(
                file_path = ["example.pdf", "fake.pdf"],
                api_key=UNSTRUCTURED_API_KEY,
                partition_via_api=True,
                chunking_strategy="by_title",
                strategy="fast",
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            1 2 0 2
            {'source': './example_data/layout-parser-paper.pdf', 'coordinates': {'points': ((16.34, 213.36), (16.34, 253.36), (36.34, 253.36), (36.34, 213.36)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'file_directory': './example_data', 'filename': 'layout-parser-paper.pdf', 'languages': ['eng'], 'last_modified': '2024-07-25T21:28:58', 'page_number': 1, 'filetype': 'application/pdf', 'category': 'UncategorizedText', 'element_id': 'd3ce55f220dfb75891b4394a18bcb973'}


    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            1 2 0 2
            {'source': './example_data/layout-parser-paper.pdf', 'coordinates': {'points': ((16.34, 213.36), (16.34, 253.36), (36.34, 253.36), (36.34, 213.36)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'file_directory': './example_data', 'filename': 'layout-parser-paper.pdf', 'languages': ['eng'], 'last_modified': '2024-07-25T21:28:58', 'page_number': 1, 'filetype': 'application/pdf', 'category': 'UncategorizedText', 'element_id': 'd3ce55f220dfb75891b4394a18bcb973'}


    Load URL:
        .. code-block:: python

            loader = UnstructuredLoader(web_url="https://www.example.com/")
            print(docs[0])

        .. code-block:: none

            page_content='Example Domain' metadata={'category_depth': 0, 'languages': ['eng'], 'filetype': 'text/html', 'url': 'https://www.example.com/', 'category': 'Title', 'element_id': 'fdaa78d856f9d143aeeed85bf23f58f8'}

        .. code-block:: python

            print(docs[1])

        .. code-block:: none

            page_content='This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.' metadata={'languages': ['eng'], 'parent_id': 'fdaa78d856f9d143aeeed85bf23f58f8', 'filetype': 'text/html', 'url': 'https://www.example.com/', 'category': 'NarrativeText', 'element_id': '3652b8458b0688639f973fe36253c992'}


    References
    ----------
    https://docs.unstructured.io/api-reference/api-services/sdk
    https://docs.unstructured.io/api-reference/api-services/overview
    https://docs.unstructured.io/open-source/core-functionality/partitioning
    https://docs.unstructured.io/open-source/core-functionality/chunking
    """  # noqa: E501

    def __init__(
        self,
        file_path: Optional[str | Path | list[str] | list[Path]] = None,
        *,
        file: Optional[IO[bytes] | list[IO[bytes]]] = None,
        partition_via_api: bool = False,
        post_processors: Optional[list[Callable[[str], str]]] = None,
        # SDK parameters
        api_key: Optional[str] = None,
        client: Optional[UnstructuredClient] = None,
        url: Optional[str] = None,
        web_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize loader."""
        if file_path is not None and file is not None:
            raise ValueError("file_path and file cannot be defined simultaneously.")
        if client is not None:
            disallowed_params = [("api_key", api_key), ("url", url)]
            bad_params = [
                param for param, value in disallowed_params if value is not None
            ]

            if bad_params:
                raise ValueError(
                    "if you are passing a custom `client`, you cannot also pass these "
                    f"params: {', '.join(bad_params)}."
                )

        unstructured_api_key = api_key or os.getenv("UNSTRUCTURED_API_KEY") or ""
        unstructured_url = url or os.getenv("UNSTRUCTURED_URL") or _DEFAULT_URL

        self.client = client or UnstructuredClient(
            api_key_auth=unstructured_api_key, server_url=unstructured_url
        )

        self.file_path = file_path
        self.file = file
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        self.unstructured_kwargs = kwargs
        if web_url:
            self.unstructured_kwargs["url"] = web_url

    def lazy_load(self) -> Iterator[Document]:
        """Load file(s) to the _UnstructuredBaseLoader."""

        def load_file(
            f: Optional[IO[bytes]] = None, f_path: Optional[str | Path] = None
        ) -> Iterator[Document]:
            """Load an individual file to the _UnstructuredBaseLoader."""
            return _SingleDocumentLoader(
                file=f,
                file_path=f_path,
                partition_via_api=self.partition_via_api,
                post_processors=self.post_processors,
                # SDK parameters
                client=self.client,
                **self.unstructured_kwargs,
            ).lazy_load()

        if isinstance(self.file, list):
            for f in self.file:
                yield from load_file(f=f)
            return

        if isinstance(self.file_path, list):
            for f_path in self.file_path:
                yield from load_file(f_path=f_path)
            return

        # Call _UnstructuredBaseLoader normally since file and file_path are not lists
        yield from load_file(f=self.file, f_path=self.file_path)


class _SingleDocumentLoader(BaseLoader):
    """Provides loader functionality for individual document/file objects.

    Encapsulates partitioning individual file objects (file or file_path) either
    locally or via the Unstructured API.
    """

    def __init__(
        self,
        file_path: Optional[str | Path] = None,
        *,
        client: UnstructuredClient,
        file: Optional[IO[bytes]] = None,
        partition_via_api: bool = False,
        post_processors: Optional[list[Callable[[str], str]]] = None,
        **kwargs: Any,
    ):
        """Initialize loader."""
        self.file_path = str(file_path) if isinstance(file_path, Path) else file_path
        self.file = file
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        # SDK parameters
        self.client = client
        self.unstructured_kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        elements_json = (
            self._post_process_elements_json(self._elements_json)
            if self.post_processors
            else self._elements_json
        )
        for element in elements_json:
            metadata = self._get_metadata()
            metadata.update(element.get("metadata"))  # type: ignore
            metadata.update(
                {"category": element.get("category") or element.get("type")}
            )
            metadata.update({"element_id": element.get("element_id")})
            yield Document(
                page_content=cast(str, element.get("text")), metadata=metadata
            )

    @property
    def _elements_json(self) -> list[dict[str, Any]]:
        """Get elements as a list of dictionaries from local partition or via API."""
        if self.partition_via_api:
            return self._elements_via_api

        return self._convert_elements_to_dicts(self._elements_via_local)

    @property
    def _elements_via_local(self) -> list[Element]:
        try:
            from unstructured.partition.auto import partition  # type: ignore
        except ImportError:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        if self.file and self.unstructured_kwargs.get("metadata_filename") is None:
            raise ValueError(
                "If partitioning a fileIO object, metadata_filename must be specified"
                " as well.",
            )

        return partition(
            file=self.file, filename=self.file_path, **self.unstructured_kwargs
        )  # type: ignore

    @property
    def _elements_via_api(self) -> list[dict[str, Any]]:
        """Retrieve a list of element dicts from the API using the SDK client."""
        client = self.client
        req = self._sdk_partition_request
        response = client.general.partition(req)  # type: ignore
        if response.status_code == 200:
            return json.loads(response.raw_response.text)
        raise ValueError(
            f"Receive unexpected status code {response.status_code} from the API.",
        )

    @property
    def _file_content(self) -> bytes:
        """Get content from either file or file_path."""
        if self.file is not None:
            return self.file.read()
        elif self.file_path:
            with open(self.file_path, "rb") as f:
                return f.read()
        raise ValueError("file or file_path must be defined.")

    @property
    def _sdk_partition_request(self) -> operations.PartitionRequest:
        return operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=self._file_content, file_name=str(self.file_path)
                ),
                **self.unstructured_kwargs,
            ),
        )

    def _convert_elements_to_dicts(
        self, elements: list[Element]
    ) -> list[dict[str, Any]]:
        return [element.to_dict() for element in elements]

    def _get_metadata(self) -> dict[str, Any]:
        """Get file_path metadata if available."""
        return {"source": self.file_path} if self.file_path else {}

    def _post_process_elements_json(
        self, elements_json: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        if self.post_processors:
            for element in elements_json:
                for post_processor in self.post_processors:
                    element["text"] = post_processor(str(element.get("text")))
        return elements_json
