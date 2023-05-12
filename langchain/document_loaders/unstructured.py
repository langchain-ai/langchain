"""Loader that uses unstructured to load files."""
import os
from abc import ABC, abstractmethod
from typing import IO, Any, List

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
        _valid_modes = {"single", "elements"}
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

    @abstractmethod
    def _get_file_mod_date(self) -> str:
        """Get file modified date and time."""

    @abstractmethod
    def _get_file_create_date(self) -> str:
        """Get file creation date and time."""

    @abstractmethod
    def _convert_unix_to_utc(self, unix_timestamp: float) -> str:
        """Convert unix timestamp to utc."""

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
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        """Initialize with file path."""
        self.file_path = file_path
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        return partition(filename=self.file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        from unstructured.file_utils.filetype import detect_filetype

        metadata = {
            "source": self.file_path,
            "filetype": detect_filetype(filename=self.file_path).name,
            "file_mod_date": self._get_file_mod_date(),
            "file_create_date": self._get_file_create_date(),
        }

        return metadata

    def _get_file_create_date(self) -> str:
        import platform

        """
        Try to get the date that a file was created. Generally possible with 
        Windows file systems. However, the same cannot be said for some UNIX 
        file systems, although most modern ones do store creation time. Even so, 
        the system call is not exposed in Python. (Anyone is welcome to write a wrapper
        around it). To avoid the trouble, we can simply fall back to modified time for 
        UNIX file systems.
        """
        if platform.system() == "Windows":
            unix_timestamp = os.path.getctime(self.file_path)
            date_time = self._convert_unix_to_utc(unix_timestamp)
            return date_time
        else:
            return self._get_file_mod_date()

    def _get_file_mod_date(self) -> str:
        unix_timestamp = os.path.getmtime(self.file_path)
        date_time = self._convert_unix_to_utc(unix_timestamp)

        return date_time

    def _convert_unix_to_utc(self, unix_timestamp: float) -> str:
        from datetime import datetime

        return datetime.utcfromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S")


class UnstructuredAPIFileLoader(UnstructuredFileLoader):
    """Loader that uses the unstructured web API to load files."""

    def __init__(
        self,
        file_path: str,
        mode: str = "single",
        url: str = "https://api.unstructured.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        min_unstructured_version = "0.6.2"
        if not satisfies_min_unstructured_version(min_unstructured_version):
            raise ValueError(
                "Partitioning via API is only supported in "
                f"unstructured>={min_unstructured_version}."
            )

        self.url = url
        self.api_key = api_key

        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.api import partition_via_api

        return partition_via_api(
            filename=self.file_path,
            api_key=self.api_key,
            api_url=self.url,
            **self.unstructured_kwargs,
        )


class UnstructuredFileIOLoader(UnstructuredBaseLoader):
    """Loader that uses unstructured to load file IO objects."""

    def __init__(self, file: IO, mode: str = "single", **unstructured_kwargs: Any):
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
        file: IO,
        mode: str = "single",
        url: str = "https://api.unstructured.io/general/v0/general",
        api_key: str = "",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""

        min_unstructured_version = "0.6.2"
        if not satisfies_min_unstructured_version(min_unstructured_version):
            raise ValueError(
                "Partitioning via API is only supported in "
                f"unstructured>={min_unstructured_version}."
            )

        self.url = url
        self.api_key = api_key
        super().__init__(file=file, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.api import partition_via_api

        return partition_via_api(
            file=self.file,
            api_key=self.api_key,
            api_url=self.url,
            **self.unstructured_kwargs,
        )
