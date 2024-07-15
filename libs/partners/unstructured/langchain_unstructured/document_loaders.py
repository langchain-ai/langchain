"""Unstructured document loader."""

import json
from pathlib import Path
from typing import IO, Any, Iterator, Sequence, Union

from langchain_community.document_loaders import UnstructuredBaseLoader
from langchain_core.documents import Document


class UnstructuredSDKFileLoader(UnstructuredBaseLoader):
    """Unstructured document loader integration.

    Load files using the `unstructured-client` sdk to the Unstructured API.

    By default, the loader makes a call to the hosted Unstructured API. If you are
    running the unstructured API locally, you can change the API rule by passing in the
    url parameter when you initialize the loader. The hosted Unstructured API requires
    an API key. See the links below to learn more about our API offerings and get an
    API key.

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
            from langchain_unstructured import UnstructuredSDKFileLoader

            loader = UnstructuredSDKFileLoader(
                # required params
                file_path = "example.pdf",
                api_key=UNSTRUCTURED_API_KEY,
                # other params
                chunking_strategy="by_page",
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
        file_path: Union[str, list[str]],
        api_key: str,
        *,
        url: str = "https://api.unstructuredapp.io/general/v0/general",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        self.file_path = file_path
        self.url = url
        self.api_key = api_key

        super().__init__(**unstructured_kwargs)

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        # This method overwrites the UnstructuredBaseLoader method because that one
        # expects `Element` objects instead of a json, which is what the SDK returns.
        elements_json = self._get_elements()
        self._post_process_elements(elements_json)

        for element in elements_json:
            metadata = self._get_metadata()
            metadata.update(element.get("metadata"))
            metadata.update(
                {"category": element.get("category") or element.get("type")}
            )
            metadata.update({"element_id": element.get("element_id")})
            yield Document(page_content=element.get("text"), metadata=metadata)

    def _get_elements(self) -> list:
        if isinstance(self.file_path, list):
            elements = []
            for path in self.file_path:
                elements.extend(
                    _get_elements_from_api(
                        file_path=path,
                        api_key=self.api_key,
                        api_url=self.url,
                        **self.unstructured_kwargs,
                    )
                )
            return elements

        return _get_elements_from_api(
            file_path=self.file_path,
            api_key=self.api_key,
            api_url=self.url,
            **self.unstructured_kwargs,
        )

    def _get_metadata(self) -> dict:
        return {"source": self.file_path}

    def _post_process_elements(self, elements: list[dict]) -> list:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element["text"] = post_processor(element.get("text"))
        return elements


class UnstructuredSDKFileIOLoader(UnstructuredBaseLoader):
    """Send file-like objects with `unstructured-client` sdk to the Unstructured API.

    By default, the loader makes a call to the hosted Unstructured API. If you are
    running the unstructured API locally, you can change the API rule by passing in the
    url parameter when you initialize the loader. The hosted Unstructured API requires
    an API key. See the links below to learn more about our API offerings and get an
    API key.

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
            from langchain_unstructured import UnstructuredSDKFileIOLoader

            with open("example.pdf", "rb") as f:
                loader = UnstructuredSDKFileIOLoader(
                    # required params
                    file_path=f,
                    api_key=UNSTRUCTURED_API_KEY,
                    # other params
                    chunking_strategy="by_page",
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
        file: Union[IO, Sequence[IO]],
        api_key: str,
        *,
        url: str = "https://api.unstructuredapp.io/general/v0/general",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        self.file = file
        self.url = url
        self.api_key = api_key

        super().__init__(**unstructured_kwargs)

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        # This method overwrites the UnstructuredBaseLoader method because that one
        # expects `Element` objects instead of a json, which is what the SDK returns.
        elements_json = self._get_elements()
        self._post_process_elements(elements_json)

        for element in elements_json:
            metadata = self._get_metadata()
            metadata.update(element.get("metadata"))
            metadata.update(
                {"category": element.get("category") or element.get("type")}
            )
            metadata.update({"element_id": element.get("element_id")})
            yield Document(page_content=element.get("text"), metadata=metadata)

    def _get_elements(self) -> list:
        if isinstance(self.file, Sequence):
            if _metadata_filenames := self.unstructured_kwargs.pop("metadata_filename"):
                elements = []
                for i, file in enumerate(self.file):
                    elements.extend(
                        _get_elements_from_api(
                            file=file,
                            file_path=_metadata_filenames[i],
                            api_key=self.api_key,
                            api_url=self.url,
                            **self.unstructured_kwargs,
                        )
                    )
                return elements
            else:
                raise ValueError(
                    "If partitioning a file via api,"
                    " metadata_filename must be specified as well.",
                )

        return _get_elements_from_api(
            file=self.file,
            file_path=self.unstructured_kwargs.pop("metadata_filename"),
            api_key=self.api_key,
            api_url=self.url,
            **self.unstructured_kwargs,
        )

    def _get_metadata(self) -> dict:
        return {}

    def _post_process_elements(self, elements: list[dict]) -> list:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element["text"] = post_processor(element.get("text"))
        return elements


def _get_elements_from_api(
    file_path: Union[str, Path],
    api_key: str,
    *,
    file: Union[IO[bytes], None] = None,
    api_url: str = "https://api.unstructuredapp.io/general/v0/general",
    **unstructured_kwargs: Any,
) -> list[dict[str, Any]]:
    """Retrieve a list of elements from the `Unstructured API` using the SDK client."""

    try:
        import unstructured_client  # noqa:F401
    except ImportError:
        raise ImportError(
            "unstructured_client package not found, please install it with "
            "`pip install unstructured-client`."
        )
    from unstructured_client.models import operations, shared

    content = _get_content(file=file, file_path=file_path)

    client = unstructured_client.UnstructuredClient(
        api_key_auth=api_key, server_url=api_url
    )
    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(content=content, file_name=str(file_path)),
            **unstructured_kwargs,
        ),
    )
    response = client.general.partition(req)

    if response.status_code == 200:
        return json.loads(response.raw_response.text)
    else:
        raise ValueError(
            f"Receive unexpected status code {response.status_code} from the API.",
        )


def _get_content(
    file_path: Union[str, Path], file: Union[IO[bytes], None] = None
) -> bytes:
    """Get content from either file or file_path."""
    # `file_path` is a required arg and used to define `file_name` for the sdk,
    # but use `file` for the `content` if it is provided
    if file is not None:
        return file.read()
    else:
        with open(file_path, "rb") as f:
            return f.read()
