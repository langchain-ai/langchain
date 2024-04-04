from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class KineticaLoader(BaseLoader):
    """Load from `Kinetica` API.

    Each document represents one row of the result. The `page_content_columns`
    are written into the `page_content` of the document. The `metadata_columns`
    are written into the `metadata` of the document. By default, all columns
    are written into the `page_content` and none into the `metadata`.

    """

    def __init__(
        self,
        query: str,
        host: str,
        username: str,
        password: str,
        parameters: Optional[Dict[str, Any]] = None,
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """Initialize Kinetica document loader.

        Args:
            query: The query to run in Kinetica.
            parameters: Optional. Parameters to pass to the query.
            page_content_columns: Optional. Columns written to Document `page_content`.
            metadata_columns: Optional. Columns written to Document `metadata`.
        """
        self.query = query
        self.host = host
        self.username = username
        self.password = password
        self.parameters = parameters
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns if metadata_columns is not None else []

    def _execute_query(self) -> List[Dict[str, Any]]:
        try:
            from gpudb import GPUdb, GPUdbSqlIterator
        except ImportError:
            raise ImportError(
                "Could not import Kinetica python API. "
                "Please install it with `pip install gpudb==7.2.0.1`."
            )

        try:
            options = GPUdb.Options()
            options.username = self.username
            options.password = self.password

            conn = GPUdb(host=self.host, options=options)

            with GPUdbSqlIterator(conn, self.query) as records:
                column_names = records.type_map.keys()
                query_result = [dict(zip(column_names, record)) for record in records]

        except Exception as e:
            print(f"An error occurred: {e}")  # noqa: T201
            query_result = []

        return query_result

    def _get_columns(
        self, query_result: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        page_content_columns = (
            self.page_content_columns if self.page_content_columns else []
        )
        metadata_columns = self.metadata_columns if self.metadata_columns else []
        if page_content_columns is None and query_result:
            page_content_columns = list(query_result[0].keys())
        if metadata_columns is None:
            metadata_columns = []
        return page_content_columns or [], metadata_columns

    def lazy_load(self) -> Iterator[Document]:
        query_result = self._execute_query()
        if isinstance(query_result, Exception):
            print(f"An error occurred during the query: {query_result}")  # noqa: T201
            return []
        page_content_columns, metadata_columns = self._get_columns(query_result)
        if "*" in page_content_columns:
            page_content_columns = list(query_result[0].keys())
        for row in query_result:
            page_content = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in page_content_columns
            )
            metadata = {k: v for k, v in row.items() if k in metadata_columns}
            doc = Document(page_content=page_content, metadata=metadata)
            yield doc

    def load(self) -> List[Document]:
        """Load data into document objects."""
        return list(self.lazy_load())
