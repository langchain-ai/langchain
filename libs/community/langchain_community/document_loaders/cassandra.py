from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.cassandra import wrapped_response_future

_NOT_SET = object()

if TYPE_CHECKING:
    from cassandra.cluster import Session
    from cassandra.pool import Host
    from cassandra.query import Statement


class CassandraLoader(BaseLoader):
    def __init__(
        self,
        table: Optional[str] = None,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        query: Union[str, Statement, None] = None,
        page_content_mapper: Callable[[Any], str] = str,
        metadata_mapper: Callable[[Any], dict] = lambda _: {},
        *,
        query_parameters: Union[dict, Sequence, None] = None,
        query_timeout: Optional[float] = _NOT_SET,  # type: ignore[assignment]
        query_trace: bool = False,
        query_custom_payload: Optional[dict] = None,
        query_execution_profile: Any = _NOT_SET,
        query_paging_state: Any = None,
        query_host: Optional[Host] = None,
        query_execute_as: Optional[str] = None,
    ) -> None:
        """
        Document Loader for Apache Cassandra.

        Args:
            table: The table to load the data from.
                (do not use together with the query parameter)
            session: The cassandra driver session.
                If not provided, the cassio resolved session will be used.
            keyspace: The keyspace of the table.
                If not provided, the cassio resolved keyspace will be used.
            query: The query used to load the data.
                (do not use together with the table parameter)
            page_content_mapper: a function to convert a row to string page content.
                Defaults to the str representation of the row.
            metadata_mapper: a function to convert a row to document metadata.
            query_parameters: The query parameters used when calling session.execute .
            query_timeout: The query timeout used when calling session.execute .
            query_trace: Whether to use tracing when calling session.execute .
            query_custom_payload: The query custom_payload used when calling
                session.execute .
            query_execution_profile: The query execution_profile used when calling
                session.execute .
            query_host: The query host used when calling session.execute .
            query_execute_as: The query execute_as used when calling session.execute .
        """
        if query and table:
            raise ValueError("Cannot specify both query and table.")

        if not query and not table:
            raise ValueError("Must specify query or table.")

        if not session or (table and not keyspace):
            try:
                from cassio.config import check_resolve_keyspace, check_resolve_session
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "Could not import a recent cassio package."
                    "Please install it with `pip install --upgrade cassio`."
                )

        if table:
            _keyspace = keyspace or check_resolve_keyspace(keyspace)
            self.query = f"SELECT * FROM {_keyspace}.{table};"
            self.metadata = {"table": table, "keyspace": _keyspace}
        else:
            self.query = query  # type: ignore[assignment]
            self.metadata = {}

        self.session = session or check_resolve_session(session)
        self.page_content_mapper = page_content_mapper
        self.metadata_mapper = metadata_mapper

        self.query_kwargs = {
            "parameters": query_parameters,
            "trace": query_trace,
            "custom_payload": query_custom_payload,
            "paging_state": query_paging_state,
            "host": query_host,
            "execute_as": query_execute_as,
        }
        if query_timeout is not _NOT_SET:
            self.query_kwargs["timeout"] = query_timeout

        if query_execution_profile is not _NOT_SET:
            self.query_kwargs["execution_profile"] = query_execution_profile

    def lazy_load(self) -> Iterator[Document]:
        for row in self.session.execute(self.query, **self.query_kwargs):
            metadata = self.metadata.copy()
            metadata.update(self.metadata_mapper(row))
            yield Document(
                page_content=self.page_content_mapper(row), metadata=metadata
            )

    async def alazy_load(self) -> AsyncIterator[Document]:
        for row in await wrapped_response_future(
            self.session.execute_async,
            self.query,
            **self.query_kwargs,
        ):
            metadata = self.metadata.copy()
            metadata.update(self.metadata_mapper(row))
            yield Document(
                page_content=self.page_content_mapper(row), metadata=metadata
            )
