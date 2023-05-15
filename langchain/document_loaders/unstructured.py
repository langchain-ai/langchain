"""Loader that uses unstructured to load files."""
from abc import ABC, abstractmethod
from typing import Any, IO, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.utils.metadata import UnstructuredMetadata


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
        self,
        file_path: str,
        mode: str = "single",
        get_source: Optional[bool] = True,
        get_created_at: Optional[bool] = True,
        get_updated_at: Optional[bool] = True,
        get_mime_type: Optional[bool] = True,
        get_extension: Optional[bool] = True,
        **unstructured_kwargs: Any,
    ):
        """Initialize arguments."""
        self.file_path = file_path
        self.get_source = get_source
        self.get_created_at = get_created_at
        self.get_updated_at = get_updated_at
        self.get_mime_type = get_mime_type
        self.get_extension = get_extension
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        return partition(filename=self.file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        # Initialise metadata dict
        metadata = {}

        # Initialise UnstructuredMetadata class object
        metadata_cls = UnstructuredMetadata(file_path=self.file_path)

        # Add required metadata to dict
        if self.get_sourcesource:
            source = metadata_cls.source()
            metadata["source"] = source
        if self.get_created_at:
            created_at = metadata_cls.created_at()
            metadata["created_at"] = created_at
        if self.get_updated_at:
            updated_at = metadata_cls.updated_at()
            metadata["updated_at"] = updated_at
        if self.get_mime_type:
            mime_type = metadata_cls.mime_type()
            metadata["mime_type"] = mime_type
        if self.get_extension:
            extension = metadata_cls.extension()
            metadata["extension"] = extension

        return metadata


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

    def __init__(
        self,
        file: IO,
        mode: str = "single",
        get_mime_type: Optional[bool] = True,
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.file = file
        self.get_mime_type = get_mime_type
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        return partition(file=self.file, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        # Initialise metadata dict
        metadata = {}

        # Initialise UnstructuredMetadata class object
        metadata_cls = UnstructuredMetadata(file_IO=self.file)

        # Add required metadata to dict
        if self.get_mime_type:
            mime_type = metadata_cls.mime_type()
            metadata["mime_type"] = mime_type

        return metadata


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
