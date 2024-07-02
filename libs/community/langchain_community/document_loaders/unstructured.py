"""Loader that uses unstructured to load files."""
from __future__ import annotations

from abc import ABC, abstractmethod
import io
import json
import logging
from pathlib import Path
from typing import IO, Any, Callable, Iterator, Optional, Sequence, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__file__)


class UnstructuredBaseLoader(BaseLoader, ABC):
    """Parent class for Unstructured Base Loaders."""

    def __init__(
        self,
        mode: str = "single",
        post_processors: Optional[list[Callable]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        # `single` - elements are combined into one (default)
        # `elements` - maintain individual elements
        # `paged` - elements are combined by page
        _valid_modes = {"single", "elements", "paged"}

        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )
        if mode=="paged":
            logger.warning("`mode='paged'` is deprecated in favor of the 'by_page' chunking strategy. Learn more about chunking here: https://docs.unstructured.io/open-source/core-functionality/chunking")
        
        self._check_if_both_mode_and_chunking_strategy_are_by_page(mode, unstructured_kwargs)

        self.mode = mode
        self.unstructured_kwargs = unstructured_kwargs
        self.post_processors = post_processors or []

    @abstractmethod
    def _get_elements(self) -> list:
        """Get elements."""

    @abstractmethod
    def _get_metadata(self) -> dict:
        """Get file_path metadata if available."""

    @abstractmethod
    def _post_process_elements(self, elements: list) -> list:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        elements = self._get_elements()
        self._post_process_elements(elements)
        if self.mode == "elements":
            for element in elements:
                metadata = self._get_metadata()
                # NOTE(MthwRobinson) - the attribute check is for backward compatibility
                # with unstructured<0.4.9. The metadata attributed was added in 0.4.9.
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                if hasattr(element, "category"):
                    metadata["category"] = element.category
                if element.to_dict().get("element_id"):
                    metadata["element_id"] = element.to_dict().get("element_id")
                yield Document(page_content=str(element), metadata=metadata)
        elif self.mode == "paged":
            text_dict: dict[int, str] = {}
            meta_dict: dict[int, dict] = {}

            for element in elements:
                metadata = self._get_metadata()
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                page_number = metadata.get("page_number", 1)

                # Check if this page_number already exists in text_dict
                if page_number not in text_dict:
                    # If not, create new entry with initial text and metadata
                    text_dict[page_number] = str(element) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    # If exists, append to text and update the metadata
                    text_dict[page_number] += str(element) + "\n\n"
                    meta_dict[page_number].update(metadata)

            # Convert the dict to a list of Document objects
            for key in text_dict.keys():
                yield Document(page_content=text_dict[key], metadata=meta_dict[key])
        elif self.mode == "single":
            metadata = self._get_metadata()
            text = "\n\n".join([str(el) for el in elements])
            yield Document(page_content=text, metadata=metadata)
        else:
            raise ValueError(f"mode of {self.mode} not supported.")

    def _check_if_both_mode_and_chunking_strategy_are_by_page(
        self, mode: str, unstructured_kwargs: dict[str, Any]
    ) -> None:
        if mode=="paged" and unstructured_kwargs.get("chunking_strategy")=="by_page":
            raise ValueError(
                "Only one of `chunking_strategy='by_page'` or `mode='paged'` may be set."
                " `chunking_strategy` is preferred."
            )


class UnstructuredFileLoader(UnstructuredBaseLoader):
    """Load files using `Unstructured`.

    The file loader uses the unstructured partition function and will automatically detect the file
    type. You can run the loader in different modes: "single", "elements", and "paged". The default
    "single" mode will return a single langchain Document object. If you use "elements" mode, the
    unstructured library will split the document into elements such as Title and NarrativeText and
    return those as individual langchain Document objects. In addition to these post-processing modes
    (which are specific to the LangChain Loaders), Unstructured has its own "chunking" parameters for
    post-processing elements into more useful chunks for uses cases such as Retrieval Augmented
    Generation (RAG). You can pass in additional unstructured kwargs to configure different
    unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredFileLoader

    loader = UnstructuredFileLoader(
        "example.pdf", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://docs.unstructured.io/open-source/core-functionality/partitioning
    https://docs.unstructured.io/open-source/core-functionality/chunking
    """

    def __init__(
        self,
        file_path: Union[str, list[str], Path, list[Path]],
        *,
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.file_path = file_path

        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        if not satisfies_min_unstructured_version("0.5.4"):
            if "strategy" in unstructured_kwargs:
                unstructured_kwargs.pop("strategy")

        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> list:
        from unstructured.partition.auto import partition

        if isinstance(self.file_path, list):
            elements = []
            for file in self.file_path:
                if isinstance(file, Path):
                    file = str(file)
                elements.extend(partition(filename=file, **self.unstructured_kwargs))
            return elements
        else:
            if isinstance(self.file_path, Path):
                self.file_path = str(self.file_path)
            return partition(filename=self.file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": self.file_path}
    
    def _post_process_elements(self, elements: list) -> list:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element.apply(post_processor)
        return elements


class UnstructuredAPIFileLoader(UnstructuredBaseLoader):
    """Load files using `unstructured-client` sdk to access the Unstructured API.

    By default, the loader makes a call to the hosted Unstructured API. If you are running the
    unstructured API locally, you can change the API rule by passing in the url parameter when you
    initialize the loader. The hosted Unstructured API requires an API key. See the links below to
    learn more about our API offerings and get an API key.

    You can run the loader in different modes: "single", "elements", and "paged". The default
    "single" mode will return a single langchain Document object. If you use "elements" mode, the
    unstructured library will split the document into elements such as Title and NarrativeText and
    return those as individual langchain Document objects. In addition to these post-processing modes
    (which are specific to the LangChain Loaders), Unstructured has its own "chunking" parameters for
    post-processing elements into more useful chunks for uses cases such as Retrieval Augmented
    Generation (RAG). You can pass in additional unstructured kwargs to configure different
    unstructured settings.

    Examples
    ```python
    from langchain_community.document_loaders import UnstructuredAPIFileLoader

    loader = UnstructuredFileAPILoader(
        "example.pdf", mode="elements", strategy="fast", api_key="MY_API_KEY",
    )
    docs = loader.load()

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
        *,
        mode: str = "single",
        url: str = "https://api.unstructured.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        self.file_path = file_path
        self.url = url
        self.api_key = api_key

        super().__init__(mode=mode, **unstructured_kwargs)

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        # This method overwrites the UnstructuredBaseLoader method because that one expects
        # `Element` objects instead of a json, which is what the SDK returns.
        elements_json = self._get_elements()
        self._post_process_elements(elements_json)

        if self.mode == "elements":
            for element in elements_json:
                metadata = self._get_metadata()
                metadata.update({"metadata": element.get("metadata")})
                metadata.update({"category": element.get("category")})
                metadata.update({"element_id": element.get("element_id")})
                yield Document(page_content=element.get("text"), metadata=metadata)
        elif self.mode == "paged":
            text_dict: dict[int, str] = {}
            meta_dict: dict[int, dict] = {}

            for element in elements_json:
                metadata = self._get_metadata()
                metadata.update({"category": element.get("category")})
                page_number = metadata.get("page_number", 1)

                # Check if this page_number already exists in text_dict
                if page_number not in text_dict:
                    # If not, create new entry with initial text and metadata
                    text_dict[page_number] = str(element.get("text")) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    # If exists, append to text and update the metadata
                    text_dict[page_number] += str(element.get("text")) + "\n\n"
                    meta_dict[page_number].update(metadata)

            # Convert the dict to a list of Document objects
            for key in text_dict.keys():
                yield Document(page_content=text_dict[key], metadata=meta_dict[key])
        elif self.mode == "single":
            metadata = self._get_metadata()
            text = "\n\n".join([el.get("text") for el in elements_json])
            yield Document(page_content=text, metadata=metadata)
        else:
            raise ValueError(f"mode of {self.mode} not supported.")

    def _get_elements(self) -> list:
        if isinstance(self.file_path, list):
            elements = []
            for path in self.file_path:
                elements.extend(
                    get_elements_from_api(
                        file_path=path,
                        api_key=self.api_key,
                        api_url=self.url,
                        **self.unstructured_kwargs,
                    )
                )
            return elements

        return get_elements_from_api(
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
    

class UnstructuredFileIOLoader(UnstructuredBaseLoader):
    """Load file-like objects opened in read mode using `Unstructured`.

    The file loader uses the unstructured partition function and will automatically detect the file
    type. You can run the loader in different modes: "single", "elements", and "paged". The default
    "single" mode will return a single langchain Document object. If you use "elements" mode, the
    unstructured library will split the document into elements such as Title and NarrativeText and
    return those as individual langchain Document objects. In addition to these post-processing modes
    (which are specific to the LangChain Loaders), Unstructured has its own "chunking" parameters for
    post-processing elements into more useful chunks for uses cases such as Retrieval Augmented
    Generation (RAG). You can pass in additional unstructured kwargs to configure different
    unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredFileIOLoader

    with open("example.pdf", "rb") as f:
        loader = UnstructuredFileIOLoader(
            f, mode="elements", strategy="fast",
        )
        docs = loader.load()


    References
    ----------
    https://docs.unstructured.io/open-source/core-functionality/partitioning
    https://docs.unstructured.io/open-source/core-functionality/chunking
    """

    def __init__(
        self,
        file: IO[bytes],
        *,
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.file = file

        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        if not satisfies_min_unstructured_version("0.5.4"):
            if "strategy" in unstructured_kwargs:
                unstructured_kwargs.pop("strategy")

        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> list:
        from unstructured.partition.auto import partition

        if isinstance(self.file, io.IOBase):
            return partition(file=self.file, **self.unstructured_kwargs)
        else:
            raise ValueError("file must be of type IO.")

    def _get_metadata(self) -> dict:
        return {}
    
    def _post_process_elements(self, elements: list) -> list:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element.apply(post_processor)
        return elements


class UnstructuredAPIFileIOLoader(UnstructuredBaseLoader):
    """Load file-like objects using the `unstructured-client` sdk to access the Unstructured API.

    By default, the loader makes a call to the hosted Unstructured API. If you are running the
    unstructured API locally, you can change the API rule by passing in the url parameter when you
    initialize the loader. The hosted Unstructured API requires an API key. See the links below to
    learn more about our API offerings and get an API key.

    You can run the loader in different modes: "single", "elements", and "paged". The default
    "single" mode will return a single langchain Document object. If you use "elements" mode, the
    unstructured library will split the document into elements such as Title and NarrativeText and
    return those as individual langchain Document objects. In addition to these post-processing modes
    (which are specific to the LangChain Loaders), Unstructured has its own "chunking" parameters
    for post-processing elements into more useful chunks for uses cases such as Retrieval Augmented
    Generation (RAG). You can pass in additional unstructured kwargs to configure different
    unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredAPIFileLoader

    with open("example.pdf", "rb") as f:
        loader = UnstructuredFileAPILoader(
            f, mode="elements", strategy="fast", api_key="MY_API_KEY",
        )
        docs = loader.load()

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
        *,
        mode: str = "single",
        url: str = "https://api.unstructured.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        self.file = file
        self.url = url
        self.api_key = api_key

        super().__init__(mode=mode, **unstructured_kwargs)

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        # This method overwrites the UnstructuredBaseLoader method because that one expects
        # `Element` objects instead of a json, which is what the SDK returns.
        elements_json = self._get_elements()
        self._post_process_elements(elements_json)

        if self.mode == "elements":
            for element in elements_json:
                metadata = self._get_metadata()
                metadata.update({"metadata": element.get("metadata")})
                metadata.update({"category": element.get("category")})
                metadata.update({"element_id": element.get("element_id")})
                yield Document(page_content=element.get("text"), metadata=metadata)
        elif self.mode == "paged":
            text_dict: dict[int, str] = {}
            meta_dict: dict[int, dict] = {}

            for element in elements_json:
                metadata = self._get_metadata()
                metadata.update({"category": element.get("category")})
                page_number = metadata.get("page_number", 1)

                # Check if this page_number already exists in text_dict
                if page_number not in text_dict:
                    # If not, create new entry with initial text and metadata
                    text_dict[page_number] = str(element.get("text")) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    # If exists, append to text and update the metadata
                    text_dict[page_number] += str(element.get("text")) + "\n\n"
                    meta_dict[page_number].update(metadata)

            # Convert the dict to a list of Document objects
            for key in text_dict.keys():
                yield Document(page_content=text_dict[key], metadata=meta_dict[key])
        elif self.mode == "single":
            metadata = self._get_metadata()
            text = "\n\n".join([el.get("text") for el in elements_json])
            yield Document(page_content=text, metadata=metadata)
        else:
            raise ValueError(f"mode of {self.mode} not supported.")
    
    def _get_elements(self) -> list:
        if isinstance(self.file, Sequence):
            if _metadata_filenames := self.unstructured_kwargs.pop("metadata_filename"):
                elements = []
                for i, file in enumerate(self.file):
                    elements.extend(
                        get_elements_from_api(
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

        return get_elements_from_api(
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


def get_elements_from_api(
    file_path: Union[str, Path],
    *,
    file: Union[IO[bytes], None] = None,
    api_url: str = "https://api.unstructured.io/general/v0/general",
    api_key: str = "",
    **unstructured_kwargs: Any,
) -> list[dict[str, Any]]:
    """Retrieve a list of elements from the `Unstructured API`."""

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
    

def _get_content(file_path: Union[str, Path], file: Union[IO[bytes], None] = None) -> bytes:
    """Get content from either file or file_path."""
    # `file_path` is a required arg and used to define `file_name` for the sdk, but use `file` for the
    # `content` if it is provided
    if file is not None:
        return file.read()
    else:
        with open(file_path, "rb") as f:
            return f.read()


def satisfies_min_unstructured_version(min_version: str) -> bool:
    """Check if the installed `Unstructured` version exceeds the minimum version
    for the feature in question."""
    from unstructured.__version__ import __version__ as __unstructured_version__

    min_version_tuple = tuple([int(x) for x in min_version.split(".")])

    # NOTE(MthwRobinson) - enables the loader to work when you're using pre-release
    # versions of unstructured like 0.4.17-dev1
    _unstructured_version = __unstructured_version__.split("-")[0]
    unstructured_version_tuple = tuple(
        [int(x) for x in _unstructured_version.split(".")]
    )

    return unstructured_version_tuple >= min_version_tuple


def validate_unstructured_version(min_unstructured_version: str) -> None:
    """Raise an error if the `Unstructured` version does not exceed the
    specified minimum."""
    if not satisfies_min_unstructured_version(min_unstructured_version):
        raise ValueError(
            f"unstructured>={min_unstructured_version} is required in this loader."
        )
