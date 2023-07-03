"""Loader that uses unstructured to load files."""
import collections
from abc import ABC, abstractmethod
from typing import IO, Any, Dict, List, Sequence, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def satisfies_min_unstructured_version(min_version: str) -> bool:
    """Checks to see if the installed unstructured version exceeds the minimum version
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
    """Raises an error if the unstructured version does not exceed the
    specified minimum."""
    if not satisfies_min_unstructured_version(min_unstructured_version):
        raise ValueError(
            f"unstructured>={min_unstructured_version} is required in this loader."
        )


class UnstructuredBaseLoader(BaseLoader, ABC):
    """Loader that uses unstructured to load files."""

    def __init__(self, mode: str = "single", **unstructured_kwargs: Any):
        """Initialize with file path."""
        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )
        _valid_modes = {"single", "elements", "paged"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )
        self.mode = mode

        if not satisfies_min_unstructured_version("0.5.4"):
            if "strategy" in unstructured_kwargs:
                unstructured_kwargs.pop("strategy")

        self.unstructured_kwargs = unstructured_kwargs

    @abstractmethod
    def _get_elements(self) -> List:
        """Get elements."""

    @abstractmethod
    def _get_metadata(self) -> dict:
        """Get metadata."""

    def load(self) -> List[Document]:
        """Load file."""
        elements = self._get_elements()
        if self.mode == "elements":
            docs: List[Document] = list()
            for element in elements:
                metadata = self._get_metadata()
                # NOTE(MthwRobinson) - the attribute check is for backward compatibility
                # with unstructured<0.4.9. The metadata attributed was added in 0.4.9.
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                if hasattr(element, "category"):
                    metadata["category"] = element.category
                docs.append(Document(page_content=str(element), metadata=metadata))
        elif self.mode == "paged":
            text_dict: Dict[int, str] = {}
            meta_dict: Dict[int, Dict] = {}

            for idx, element in enumerate(elements):
                metadata = self._get_metadata()
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                page_number = metadata.get("page_number", 1)

                # Check if this page_number already exists in docs_dict
                if page_number not in text_dict:
                    # If not, create new entry with initial text and metadata
                    text_dict[page_number] = str(element) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    # If exists, append to text and update the metadata
                    text_dict[page_number] += str(element) + "\n\n"
                    meta_dict[page_number].update(metadata)

            # Convert the dict to a list of Document objects
            docs = [
                Document(page_content=text_dict[key], metadata=meta_dict[key])
                for key in text_dict.keys()
            ]
        elif self.mode == "single":
            metadata = self._get_metadata()
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(page_content=text, metadata=metadata)]
        else:
            raise ValueError(f"mode of {self.mode} not supported.")
        return docs


class UnstructuredFileLoader(UnstructuredBaseLoader):
    """Loader that uses unstructured to load files."""

    def __init__(
        self,
        file_path: Union[str, List[str]],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.file_path = file_path
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        return partition(filename=self.file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": self.file_path}


def get_elements_from_api(
    file_path: Union[str, List[str], None] = None,
    file: Union[IO, Sequence[IO], None] = None,
    api_url: str = "https://api.unstructured.io/general/v0/general",
    api_key: str = "",
    **unstructured_kwargs: Any,
) -> List:
    """Retrieves a list of elements from the Unstructured API."""
    if isinstance(file, collections.abc.Sequence) or isinstance(file_path, list):
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
            filename=file_path,
            file=file,
            api_key=api_key,
            api_url=api_url,
            **unstructured_kwargs,
        )


class UnstructuredAPIFileLoader(UnstructuredFileLoader):
    """Loader that uses the unstructured web API to load files."""

    def __init__(
        self,
        file_path: Union[str, List[str]] = "",
        mode: str = "single",
        url: str = "https://api.unstructured.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        if isinstance(file_path, str):
            validate_unstructured_version(min_unstructured_version="0.6.2")
        else:
            validate_unstructured_version(min_unstructured_version="0.6.3")

        self.url = url
        self.api_key = api_key

        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": self.file_path}

    def _get_elements(self) -> List:
        return get_elements_from_api(
            file_path=self.file_path,
            api_key=self.api_key,
            api_url=self.url,
            **self.unstructured_kwargs,
        )


class UnstructuredFileIOLoader(UnstructuredBaseLoader):
    """Loader that uses unstructured to load file IO objects."""

    def __init__(
        self,
        file: Union[IO, Sequence[IO]],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.file = file
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        return partition(file=self.file, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {}


class UnstructuredAPIFileIOLoader(UnstructuredFileIOLoader):
    """Loader that uses the unstructured web API to load file IO objects."""

    def __init__(
        self,
        file: Union[IO, Sequence[IO]],
        mode: str = "single",
        url: str = "https://api.unstructured.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        if isinstance(file, collections.abc.Sequence):
            validate_unstructured_version(min_unstructured_version="0.6.3")
        if file:
            validate_unstructured_version(min_unstructured_version="0.6.2")

        self.url = url
        self.api_key = api_key

        super().__init__(file=file, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        return get_elements_from_api(
            file=self.file,
            api_key=self.api_key,
            api_url=self.url,
            **self.unstructured_kwargs,
        )
