from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import teradatasql
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class TeradataLoader(BaseLoader):
    """Load documents from a Teradata database using the teradatasql driver.

    Each document represents one row of the result. The `page_content_columns`
    are written into the `page_content` of the document. The `metadata_columns`
    are written into the `metadata` of the document. By default, all columns
    are written into the `page_content` and none into the `metadata`.
    """

    def __init__(
        self,
        query: str,
        db_url: str,
        user: str,
        password: str,
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """Initialize the TeradataLoader with connection details and query specifics.
        Args:
            query (str): SQL query to execute.
            db_url (str): URL of the Teradata database.
            user (str): Username for authentication.
            password (str): Password for authentication.
            page_content_columns: Optional. Columns written to Document `page_content`.
            metadata_columns: Optional. Columns to include in document metadata.
        """
        self.query = query
        self.db_url = db_url
        self.user = user
        self.password = password
        self.page_content_columns = page_content_columns or [
            "*"
        ]  # if not provided all columns are content
        self.metadata_columns = (
            metadata_columns or []
        )  # if none is provided metadata will be empty

    def _execute_query(self) -> List[Dict[str, Any]]:
        """Executes the SQL query and returns a list of dictionaries."""
        with teradatasql.connect(
            host=self.db_url, user=self.user, password=self.password
        ) as con:
            with con.cursor() as cur:
                cur.execute(self.query)
                column_names = [desc[0] for desc in cur.description]
                return [dict(zip(column_names, row)) for row in cur.fetchall()]

    def _get_columns(
        self, query_result: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Determines the columns for page content and metadata."""
        if "*" in self.page_content_columns and query_result:
            page_content_columns = list(query_result[0].keys())
        else:
            page_content_columns = self.page_content_columns
        metadata_columns = self.metadata_columns
        return page_content_columns, metadata_columns

    def lazy_load(self) -> Iterator[Document]:
        """Lazily loads documents from the query results."""
        query_result = self._execute_query()
        page_content_columns, metadata_columns = self._get_columns(query_result)
        for row in query_result:
            page_content = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in page_content_columns
            )
            metadata = {k: v for k, v in row.items() if k in metadata_columns}
            yield Document(page_content=page_content, metadata=metadata)
