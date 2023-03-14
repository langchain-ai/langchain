"""Loader that uses unstructured to load files."""
from abc import ABC, abstractmethod
from typing import IO, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def satisfies_min_unstructured_version(min_version: str):
    """Checks to see if the installed unstructured version exceeds the minimum version
    for the feature in question."""
    from unstructured.__version__ import __version__ as __unstructured_version__

    min_version_tuple = tuple(min_version.split("."))

    # NOTE(MthwRobinson) - enables the loader to work when you're using pre-release
    # versions of unstructured like 0.4.17-dev1
    _unstructured_version = __unstructured_version__.split("-")[0]
    unstructured_version_tuple = tuple(
        [int(x) for x in _unstructured_version.split(".")]
    )

    return min_version_tuple >= unstructured_version_tuple


class UnstructuredBaseLoader(BaseLoader, ABC):
    """Loader that uses unstructured to load files."""

    def __init__(self, mode: str = "single", strategy: str = "hi_res"):
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
        self.strategy = strategy

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

    def __init__(self, file_path: str, mode: str = "single", strategy: str = "hi_res"):
        """Initialize with file path."""
        self.file_path = file_path
        super().__init__(mode=mode, strategy=strategy)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        kwargs = {"filename": self.file_path}
        if satisfies_min_unstructured_version("0.5.4"):
            kwargs["strategy"] = self.strategy

        return partition(**kwargs)

    def _get_metadata(self) -> dict:
        return {"source": self.file_path}


class UnstructuredFileIOLoader(UnstructuredBaseLoader):
    """Loader that uses unstructured to load file IO objects."""

    def __init__(self, file: IO, mode: str = "single"):
        """Initialize with file path."""
        self.file = file
        super().__init__(mode=mode)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        kwargs = {"filename": self.file_path}
        if satisfies_min_unstructured_version("0.5.4"):
            kwargs["strategy"] = self.strategy

        return partition(**kwargs)

    def _get_metadata(self) -> dict:
        return {}
