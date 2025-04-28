"""Loader that uses unstructured to load files."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Iterator, List, Optional, Sequence, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from typing_extensions import TypeAlias

from langchain_community.document_loaders.base import BaseLoader

Element: TypeAlias = Any

logger = logging.getLogger(__file__)


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


class UnstructuredBaseLoader(BaseLoader, ABC):
    """Base Loader that uses `Unstructured`."""

    def __init__(
        self,
        mode: str = "single",  # deprecated
        post_processors: Optional[List[Callable[[str], str]]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        # `single` - elements are combined into one (default)
        # `elements` - maintain individual elements
        # `paged` - elements are combined by page
        _valid_modes = {"single", "elements", "paged"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )

        if not satisfies_min_unstructured_version("0.5.4"):
            if "strategy" in unstructured_kwargs:
                unstructured_kwargs.pop("strategy")

        self._check_if_both_mode_and_chunking_strategy_are_by_page(
            mode, unstructured_kwargs
        )
        self.mode = mode
        self.unstructured_kwargs = unstructured_kwargs
        self.post_processors = post_processors or []

    @abstractmethod
    def _get_elements(self) -> List[Element]:
        """Get elements."""

    @abstractmethod
    def _get_metadata(self) -> dict[str, Any]:
        """Get file_path metadata if available."""

    def _post_process_elements(self, elements: List[Element]) -> List[Element]:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element.apply(post_processor)
        return elements

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
            logger.warning(
                "`mode='paged'` is deprecated in favor of the 'by_page' chunking"
                " strategy. Learn more about chunking here:"
                " https://docs.unstructured.io/open-source/core-functionality/chunking"
            )
            text_dict: dict[int, str] = {}
            meta_dict: dict[int, dict[str, Any]] = {}

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
        if (
            mode == "paged"
            and unstructured_kwargs.get("chunking_strategy") == "by_page"
        ):
            raise ValueError(
                "Only one of `chunking_strategy='by_page'` or `mode='paged'` may be"
                " set. `chunking_strategy` is preferred."
            )


@deprecated(
    since="0.2.8",
    removal="1.0",
    alternative_import="langchain_unstructured.UnstructuredLoader",
)
class UnstructuredFileLoader(UnstructuredBaseLoader):
    """Load files using `Unstructured`.

    The file loader uses the unstructured partition function and will automatically
    detect the file type. You can run the loader in different modes: "single",
    "elements", and "paged". The default "single" mode will return a single langchain
    Document object. If you use "elements" mode, the unstructured library will split
    the document into elements such as Title and NarrativeText and return those as
    individual langchain Document objects. In addition to these post-processing modes
    (which are specific to the LangChain Loaders), Unstructured has its own "chunking"
    parameters for post-processing elements into more useful chunks for uses cases such
    as Retrieval Augmented Generation (RAG). You can pass in additional unstructured
    kwargs to configure different unstructured settings.

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
        file_path: Union[str, List[str], Path, List[Path]],
        *,
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.file_path = file_path

        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List[Element]:
        from unstructured.partition.auto import partition

        if isinstance(self.file_path, list):
            elements: List[Element] = []
            for file in self.file_path:
                if isinstance(file, Path):
                    file = str(file)
                elements.extend(partition(filename=file, **self.unstructured_kwargs))
            return elements
        else:
            if isinstance(self.file_path, Path):
                self.file_path = str(self.file_path)
            return partition(filename=self.file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict[str, Any]:
        return {"source": self.file_path}


def get_elements_from_api(
    file_path: Union[str, List[str], Path, List[Path], None] = None,
    file: Union[IO[bytes], Sequence[IO[bytes]], None] = None,
    api_url: str = "https://api.unstructuredapp.io/general/v0/general",
    api_key: str = "",
    **unstructured_kwargs: Any,
) -> List[Element]:
    """Retrieve a list of elements from the `Unstructured API`."""
    if is_list := isinstance(file_path, list):
        file_path = [str(path) for path in file_path]
    if isinstance(file, Sequence) or is_list:
        from unstructured.partition.api import partition_multiple_via_api

        _doc_elements = partition_multiple_via_api(
            filenames=file_path,
            files=file,
            api_key=api_key,
            api_url=api_url,
            **unstructured_kwargs,
        )
        elements = []
        for _elements in _doc_elements:
            elements.extend(_elements)
        return elements
    else:
        from unstructured.partition.api import partition_via_api

        return partition_via_api(
            filename=str(file_path) if file_path is not None else None,
            file=file,
            api_key=api_key,
            api_url=api_url,
            **unstructured_kwargs,
        )


@deprecated(
    since="0.2.8",
    removal="1.0",
    alternative_import="langchain_unstructured.UnstructuredLoader",
)
class UnstructuredAPIFileLoader(UnstructuredBaseLoader):
    """Load files using `Unstructured` API.

    By default, the loader makes a call to the hosted Unstructured API. If you are
    running the unstructured API locally, you can change the API rule by passing in the
    url parameter when you initialize the loader. The hosted Unstructured API requires
    an API key. See the links below to learn more about our API offerings and get an
    API key.

    You can run the loader in different modes: "single", "elements", and "paged". The
    default "single" mode will return a single langchain Document object. If you use
    "elements" mode, the unstructured library will split the document into elements such
    as Title and NarrativeText and return those as individual langchain Document
    objects. In addition to these post-processing modes (which are specific to the
    LangChain Loaders), Unstructured has its own "chunking" parameters for
    post-processing elements into more useful chunks for uses cases such as Retrieval
    Augmented Generation (RAG). You can pass in additional unstructured kwargs to
    configure different unstructured settings.

    Examples
    ```python
    from langchain_community.document_loaders import UnstructuredAPIFileLoader

    loader = UnstructuredAPIFileLoader(
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
        file_path: Union[str, List[str]],
        *,
        mode: str = "single",
        url: str = "https://api.unstructuredapp.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        validate_unstructured_version(min_unstructured_version="0.10.15")

        self.file_path = file_path
        self.url = url
        self.api_key = os.getenv("UNSTRUCTURED_API_KEY") or api_key

        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_metadata(self) -> dict[str, Any]:
        return {"source": self.file_path}

    def _get_elements(self) -> List[Element]:
        return get_elements_from_api(
            file_path=self.file_path,
            api_key=self.api_key,
            api_url=self.url,
            **self.unstructured_kwargs,
        )

    def _post_process_elements(self, elements: List[Element]) -> List[Element]:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element.apply(post_processor)
        return elements


@deprecated(
    since="0.2.8",
    removal="1.0",
    alternative_import="langchain_unstructured.UnstructuredLoader",
)
class UnstructuredFileIOLoader(UnstructuredBaseLoader):
    """Load file-like objects opened in read mode using `Unstructured`.

    The file loader uses the unstructured partition function and will automatically
    detect the file type. You can run the loader in different modes: "single",
    "elements", and "paged". The default "single" mode will return a single langchain
    Document object. If you use "elements" mode, the unstructured library will split
    the document into elements such as Title and NarrativeText and return those as
    individual langchain Document objects. In addition to these post-processing modes
    (which are specific to the LangChain Loaders), Unstructured has its own "chunking"
    parameters for post-processing elements into more useful chunks for uses cases
    such as Retrieval Augmented Generation (RAG). You can pass in additional
    unstructured kwargs to configure different unstructured settings.

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
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List[Element]:
        from unstructured.partition.auto import partition

        return partition(file=self.file, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict[str, Any]:
        return {}

    def _post_process_elements(self, elements: List[Element]) -> List[Element]:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element.apply(post_processor)
        return elements


@deprecated(
    since="0.2.8",
    removal="1.0",
    alternative_import="langchain_unstructured.UnstructuredLoader",
)
class UnstructuredAPIFileIOLoader(UnstructuredBaseLoader):
    """Send file-like objects with `unstructured-client` sdk to the Unstructured API.

    By default, the loader makes a call to the hosted Unstructured API. If you are
    running the unstructured API locally, you can change the API rule by passing in the
    url parameter when you initialize the loader. The hosted Unstructured API requires
    an API key. See the links below to learn more about our API offerings and get an
    API key.

    You can run the loader in different modes: "single", "elements", and "paged". The
    default "single" mode will return a single langchain Document object. If you use
    "elements" mode, the unstructured library will split the document into elements
    such as Title and NarrativeText and return those as individual langchain Document
    objects. In addition to these post-processing modes (which are specific to the
    LangChain Loaders), Unstructured has its own "chunking" parameters for
    post-processing elements into more useful chunks for uses cases such as Retrieval
    Augmented Generation (RAG). You can pass in additional unstructured kwargs to
    configure different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredAPIFileLoader

    with open("example.pdf", "rb") as f:
        loader = UnstructuredAPIFileIOLoader(
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
        file: Union[IO[bytes], Sequence[IO[bytes]]],
        *,
        mode: str = "single",
        url: str = "https://api.unstructuredapp.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        if isinstance(file, Sequence):
            validate_unstructured_version(min_unstructured_version="0.6.3")
        validate_unstructured_version(min_unstructured_version="0.6.2")

        self.file = file
        self.url = url
        self.api_key = os.getenv("UNSTRUCTURED_API_KEY") or api_key

        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List[Element]:
        if self.unstructured_kwargs.get("metadata_filename"):
            return get_elements_from_api(
                file=self.file,
                file_path=self.unstructured_kwargs.pop("metadata_filename"),
                api_key=self.api_key,
                api_url=self.url,
                **self.unstructured_kwargs,
            )
        else:
            raise ValueError(
                "If partitioning a file via api,"
                " metadata_filename must be specified as well.",
            )

    def _get_metadata(self) -> dict[str, Any]:
        return {}

    def _post_process_elements(self, elements: List[Element]) -> List[Element]:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        for element in elements:
            for post_processor in self.post_processors:
                element.apply(post_processor)
        return elements
