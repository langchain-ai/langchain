import logging
from typing import Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class CouchbaseLoader(BaseLoader):
    """Load documents from `Couchbase`.

    Each document represents one row of the result. The `page_content_fields` are
    written into the `page_content`of the document. The `metadata_fields` are written
    into the `metadata` of the document. By default, all columns are written into
    the `page_content` and none into the `metadata`.
    """

    def __init__(
        self,
        connection_string: str,
        db_username: str,
        db_password: str,
        query: str,
        *,
        page_content_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """Initialize Couchbase document loader.

        Args:
            connection_string (str): The connection string to the Couchbase cluster.
            db_username (str): The username to connect to the Couchbase cluster.
            db_password (str): The password to connect to the Couchbase cluster.
            query (str): The SQL++ query to execute.
            page_content_fields (Optional[List[str]]): The columns to write into the
                `page_content` field of the document. By default, all columns are
                written.
            metadata_fields (Optional[List[str]]): The columns to write into the
                `metadata` field of the document. By default, no columns are written.
        """
        try:
            from couchbase.auth import PasswordAuthenticator
            from couchbase.cluster import Cluster
            from couchbase.options import ClusterOptions
        except ImportError as e:
            raise ImportError(
                "Could not import couchbase package."
                "Please install couchbase SDK with `pip install couchbase`."
            ) from e
        if not connection_string:
            raise ValueError("connection_string must be provided.")

        if not db_username:
            raise ValueError("db_username must be provided.")

        if not db_password:
            raise ValueError("db_password must be provided.")

        auth = PasswordAuthenticator(
            db_username,
            db_password,
        )

        self.cluster: Cluster = Cluster(connection_string, ClusterOptions(auth))
        self.query = query
        self.page_content_fields = page_content_fields
        self.metadata_fields = metadata_fields

    def lazy_load(self) -> Iterator[Document]:
        """Load Couchbase data into Document objects lazily."""
        from datetime import timedelta

        # Ensure connection to Couchbase cluster
        self.cluster.wait_until_ready(timedelta(seconds=5))

        # Run SQL++ Query
        result = self.cluster.query(self.query)
        for row in result:
            metadata_fields = self.metadata_fields
            page_content_fields = self.page_content_fields

            if not page_content_fields:
                page_content_fields = list(row.keys())

            if not metadata_fields:
                metadata_fields = []

            metadata = {field: row[field] for field in metadata_fields}

            document = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in page_content_fields
            )

            yield (Document(page_content=document, metadata=metadata))
