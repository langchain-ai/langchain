"""Unstructured document loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Iterator, Optional, cast

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_unstructured.utils import lazyproperty

logger = logging.getLogger(__file__)

if TYPE_CHECKING:
    import requests
    from unstructured.documents.elements import Element
    from unstructured_client.utils import RetryConfig


class UnstructuredLoader(BaseLoader):
    """Unstructured document loader interface.

    Partition and load files using either the `unstructured-client` sdk and the
    Unstructured API or locally using the `unstructured` library.

    API:
    To partition via the Unstructured API `pip install unstructured-client` and set
    `partition_via_api=True` and define `api_key`. If you are running the unstructured
    API locally, you can change the API rule by defining `url` when you initialize the
    loader. The hosted Unstructured API requires an API key. See the links below to
    learn more about our API offerings and get an API key.

    Local:
    By default the file loader uses the Unstructured `partition` function and will
    automatically detect the file type.

    In addition to document specific partition parameters, Unstructured has a rich set
    of "chunking" parameters for post-processing elements into more useful text segments
    for uses cases such as Retrieval Augmented Generation (RAG). You can pass additional
    Unstructured kwargs to the loader to configure different unstructured settings.

    Setup:
        .. code-block:: bash
            pip install -U langchain-unstructured
            pip install -U unstructured-client
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

    Load:
        .. code-block:: python
            docs = loader.load()

            print(docs[0].page_content[:100])
            print(docs[0].metadata)

    References
    ----------
    https://docs.unstructured.io/api-reference/api-services/sdk
    https://docs.unstructured.io/api-reference/api-services/overview
    https://docs.unstructured.io/open-source/core-functionality/partitioning
    https://docs.unstructured.io/open-source/core-functionality/chunking
    """

    def __init__(
        self,
        file_path: Optional[str | Path | list[str] | list[Path]] = None,
        *,
        file: Optional[IO[bytes] | list[IO[bytes]]] = None,
        partition_via_api: bool = False,
        post_processors: Optional[list[Callable[[str], str]]] = None,
        # SDK parameters
        api_key: Optional[str] = None,
        client: Optional[requests.Session] = None,
        retry_config: Optional[RetryConfig] = None,
        server: Optional[str] = None,
        url: Optional[str] = "https://api.unstructuredapp.io/general/v0/general",
        url_params: Optional[dict[str, str]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize loader."""
        self.file_path = file_path
        self.file = file
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        # SDK parameters
        self.api_key = api_key
        self.client = client
        self.retry_config = retry_config
        self.server = server
        self.url = url
        self.url_params = url_params
        self.unstructured_kwargs = unstructured_kwargs

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
                api_key=self.api_key,
                client=self.client,
                retry_config=self.retry_config,
                server=self.server,
                url=self.url,
                url_params=self.url_params,
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
        file: Optional[IO[bytes]] = None,
        partition_via_api: bool = False,
        post_processors: Optional[list[Callable[[str], str]]] = None,
        # SDK parameters
        api_key: Optional[str] = None,
        client: Optional[requests.Session] = None,
        retry_config: Optional[RetryConfig] = None,
        server: Optional[str] = None,
        url: Optional[str] = "https://api.unstructuredapp.io/general/v0/general",
        url_params: Optional[dict[str, str]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize loader."""
        self.file_path = str(file_path) if isinstance(file_path, Path) else file_path
        self.file = file
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        # SDK parameters
        self.api_key = api_key
        self.client = client
        self.retry_config = retry_config
        self.server = server
        self.url = url
        self.url_params = url_params
        self.unstructured_kwargs = unstructured_kwargs

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

    @lazyproperty
    def _elements_json(self) -> list[dict[str, Any]]:
        """Get elements as a list of dictionaries from local partition or via API."""
        if self.partition_via_api and self.api_key is None:
            raise ValueError(
                "If partitioning via the API, api_key must be defined.",
            )
        if not self.partition_via_api and self.api_key:
            logger.warning(
                "Partitioning locally even though api_key is defined since"
                " partition_via_api=False.",
            )
        if self.partition_via_api and self.api_key:
            return self._elements_via_api

        return self._convert_elements_to_dicts(self._elements_via_local)

    @lazyproperty
    def _elements_via_local(self) -> list[Element]:
        try:
            from unstructured.partition.auto import partition  # noqa:F401
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
        )

    @lazyproperty
    def _elements_via_api(self) -> list[dict[str, Any]]:
        """Retrieve a list of element dicts from the API using the SDK client."""
        client = self._sdk_client
        req = self._sdk_partition_request
        response = client.general.partition(req)
        if response.status_code == 200:
            return json.loads(response.raw_response.text)
        raise ValueError(
            f"Receive unexpected status code {response.status_code} from the API.",
        )

    @lazyproperty
    def _file_content(self) -> bytes:
        """Get content from either file or file_path."""
        if self.file is not None:
            return self.file.read()
        elif self.file_path:
            with open(self.file_path, "rb") as f:
                return f.read()
        raise ValueError("file or file_path must be defined.")

    @lazyproperty
    def _sdk_client(self):
        try:
            import unstructured_client  # noqa:F401
        except ImportError:
            raise ImportError(
                "unstructured_client package not found, please install it with"
                " `pip install unstructured-client`."
            )
        return unstructured_client.UnstructuredClient(
            api_key_auth=self.api_key,  # type: ignore
            client=self.client,
            retry_config=self.retry_config,
            server=self.server,
            server_url=self.url,
            url_params=self.url_params,
        )

    @lazyproperty
    def _sdk_partition_request(self):
        try:
            from unstructured_client.models import operations, shared
        except ImportError:
            raise ImportError(
                "unstructured_client package not found, please install it with"
                " `pip install unstructured-client`."
            )
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
        for element in elements_json:
            for post_processor in self.post_processors:  # type: ignore
                element["text"] = post_processor(str(element.get("text")))
        return elements_json
