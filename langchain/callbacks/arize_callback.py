"""
Callback handlers to log to Arize and Phoenix-compatible file formats.
"""

import importlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypeGuard,
    Union,
)
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.document import Document as LangChainDocument

if TYPE_CHECKING:
    import pandas as pd
    from arize.api import Client as ArizeClient
    from phoenix import Dataset as PhoenixDataset
    from phoenix import Schema as PhoenixSchema
    from pyarrow.fs import FileSelector, FileSystem

logger = logging.getLogger(__name__)


Embedding: TypeAlias = List[float]
PathType: TypeAlias = Union[str, bytes, os.PathLike]


@dataclass
class ChainData(ABC):
    ...


@dataclass
class RetrievalAugmentedGenerationData(ChainData):
    query_text: Optional[str] = None
    query_embedding: Optional[Embedding] = None
    response_text: Optional[str] = None
    document_ids: Optional[List[UUID]] = None
    scores: Optional[List[float]] = None


def is_retrieval_qa_chain_type(chain_name: str) -> bool:
    return str(chain_name).endswith("RetrievalQA")


def is_retrieval_augmented_generation_data(
    chain_data: Any,
) -> TypeGuard[RetrievalAugmentedGenerationData]:
    return isinstance(chain_data, RetrievalAugmentedGenerationData)


def get_chain_data_type(base_chain_name: str) -> Type[ChainData]:
    # if is_retrieval_qa_chain_type(chain_type):
    return RetrievalAugmentedGenerationData
    # raise ValueError(f"Unsupported chain type: {chain_type}.")


def import_arize_client() -> "ArizeClient":
    """
    Imports arize.api.Client and raises an error if the arize package is not installed.
    """

    try:
        from arize.api import Client
    except ImportError:
        raise ImportError(
            "The arize package is not installed. Run `pip install arize`."
        )
    return Client


def _import_package(package_name: str) -> ModuleType:
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        raise ImportError(f"The {package_name} package must be installed.")
    return package


def validate_package_version(
    package: ModuleType,
    min_supported_version: Optional[str],
    max_supported_version: Optional[str],
) -> None:
    # TODO: implement package version validation at runtime
    ...


class BaseArizeCallbackHandler(BaseCallbackHandler, ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._chain_data: Optional[ChainData] = None
        self._base_chain_name: Optional[str] = None

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        chain_name = ".".join(serialized["id"])
        if self._is_base_chain(parent_run_id):
            if not is_retrieval_qa_chain_type(chain_name):
                raise ValueError(f"Unsupported chain type: {chain_name}.")
            self._base_chain_name = chain_name
            chain_data_type = get_chain_data_type(base_chain_name=chain_name)
            self._chain_data = chain_data_type()
        self._add_to_chain_data_on_chain_start(chain_name, inputs)

    def _add_to_chain_data_on_chain_start(
        self,
        chain_name: str,
        inputs: Dict[str, Any],
    ) -> None:
        if self._chain_data is None:
            raise ValueError("Chain data must be initialized before adding to it.")
        if is_retrieval_qa_chain_type(
            chain_name
        ) and is_retrieval_augmented_generation_data(self._chain_data):
            self._chain_data.query_text = inputs["query"]

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self._is_base_chain(parent_run_id) and self._chain_data is not None:
            self._add_to_chain_data_on_chain_end(outputs)
            self._log_chain_data(self._chain_data)
            self._chain_data = None

    def _add_to_chain_data_on_chain_end(self, output: Dict[str, Any]) -> None:
        if is_retrieval_augmented_generation_data(self._chain_data):
            self._chain_data.response_text = output["result"]

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        ...

    def on_retriever_end(
        self,
        documents: Sequence[LangChainDocument],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if is_retrieval_augmented_generation_data(self._chain_data):
            # TODO: we need document ids and cosine similarity scores here
            pass

    @abstractmethod
    def _log_chain_data(self, chain_data: ChainData) -> None:
        ...

    @staticmethod
    def _is_base_chain(parent_run_id: Optional[UUID]) -> bool:
        return parent_run_id is None


class ArizeCallbackHandler(BaseArizeCallbackHandler):
    arize = _import_package("arize")

    def __init__(
        self,
        arize_client: "ArizeClient",
        model_id: str,
        model_version: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        validate_package_version(self.arize, None, None)
        super().__init__(**kwargs)
        self._arize_client = arize_client
        self._model_id = model_id
        self._model_version = model_version

    @classmethod
    def from_credentials(
        cls,
        api_key: str,
        space_key: str,
        model_id: str,
        model_version: str,
        arize_client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "ArizeCallbackHandler":
        ArizeClient = import_arize_client()
        arize_client = ArizeClient(
            space_key=space_key,
            api_key=api_key,
            **(arize_client_kwargs or {}),
        )
        return cls(arize_client, model_id, model_version, **kwargs)

    def _log_chain_data(self, chain_data: ChainData) -> None:
        # TODO: implement logging of chain data to arize
        print(self._chain_data)


class PhoenixCallbackHandler(BaseArizeCallbackHandler):
    pandas = _import_package("pandas")
    phoenix = _import_package("phoenix")
    pyarrow = _import_package("pyarrow")

    def __init__(
        self,
        file_system: Optional["FileSystem"],
        data_path_or_selector: Optional[Union[PathType, "FileSelector"]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes a PhoenixCallbackHandler.

        Args:
            file_system (Optional[FileSystem]): An instance of
            pyarrow.fs.FileSystem, or None. If no file system is provided, data
            will be stored in memory.

            data_path_or_selector (Optional[Union[PathType,
            FileSelector]]): A path to a directory or a pyarrow.fs.FileSelector
            object. If no path is provided, a temporary directory will be
            created. If no file_system parameter is provided, this parameter
            should be None.
        """
        self._file_system = file_system
        self._data_path_or_selector = data_path_or_selector
        super().__init__(*args, **kwargs)

    @classmethod
    def to_local_filesystem(
        cls,
        data_path: PathType,
    ) -> "PhoenixCallbackHandler":
        """Returns a PhoenixCallbackHandler that logs data to the local file
        system.

        Args:
            data_path (PathType): A path to a directory where data will be
            logged. The directory may contain existing data. If the directory
            does not exist, it will be created.

        Returns:
            PhoenixCallbackHandler: A callback handler that logs data to the
            local file system.
        """
        local_file_system = cls.pyarrow.fs.LocalFileSystem()
        return cls(
            file_system=local_file_system,
            data_path_or_selector=data_path,
        )

    def _log_chain_data(self, chain_data: ChainData) -> None:
        # TODO: implement logging of chain data to phoenix-compatible file format
        ...

    @property
    def schema(self) -> "PhoenixSchema":
        """The Phoenix schema of the logged data.

        Returns:
            phoenix.Schema: An schema describing the logged data.
        """
        raise NotImplementedError

    @property
    def dataframe(self) -> "pd.DataFrame":
        """A pandas DataFrame containing the logged data.

        Returns:
            pandas.DataFrame: A dataframe containing the logged data.
        """
        raise NotImplementedError

    @property
    def dataset(self) -> "PhoenixDataset":
        """A Phoenix dataset containing the logged data.

        Returns:
            phoenix.Dataset: A dataset containing the logged data.
        """

        raise NotImplementedError
