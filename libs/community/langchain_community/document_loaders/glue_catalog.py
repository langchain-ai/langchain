from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    from boto3.session import Session


class GlueCatalogLoader(BaseLoader):
    """Load table schemas from AWS Glue.

    This loader fetches the schema of each table within a specified AWS Glue database.
    The schema details include column names and their data types, similar to pandas
    dtype representation.

    AWS credentials are automatically loaded using boto3, following the standard AWS
    method:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific AWS profile is required, it can be specified and will be used to
    establish the session.
    """

    def __init__(
        self,
        database: str,
        *,
        session: Optional[Session] = None,
        profile_name: Optional[str] = None,
        table_filter: Optional[List[str]] = None,
    ):
        """Initialize Glue database loader.

        Args:
            database: The name of the Glue database from which to load table schemas.
            session: Optional. A boto3 Session object. If not provided, a new
                session will be created.
            profile_name: Optional. The name of the AWS profile to use for credentials.
            table_filter: Optional. List of table names to fetch schemas for,
                fetching all if None.
        """
        self.database = database
        self.profile_name = profile_name
        self.table_filter = table_filter
        if session:
            self.glue_client = session.client("glue")
        else:
            self.glue_client = self._initialize_glue_client()

    def _initialize_glue_client(self) -> Any:
        """Initialize the AWS Glue client.

        Returns:
            The initialized AWS Glue client.

        Raises:
            ValueError: If there is an issue with AWS session/client initialization.
        """
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required to use the GlueCatalogLoader. "
                "Please install it with `pip install boto3`."
            ) from e

        try:
            session = (
                boto3.Session(profile_name=self.profile_name)
                if self.profile_name
                else boto3.Session()
            )
            return session.client("glue")
        except Exception as e:
            raise ValueError("Issue with AWS session/client initialization.") from e

    def _fetch_tables(self) -> List[str]:
        """Retrieve all table names in the specified Glue database.

        Returns:
            A list of table names.
        """
        paginator = self.glue_client.get_paginator("get_tables")
        table_names = []
        for page in paginator.paginate(DatabaseName=self.database):
            for table in page["TableList"]:
                if self.table_filter is None or table["Name"] in self.table_filter:
                    table_names.append(table["Name"])
        return table_names

    def _fetch_table_schema(self, table_name: str) -> Dict[str, str]:
        """Fetch the schema of a specified table.

        Args:
            table_name: The name of the table for which to fetch the schema.

        Returns:
            A dictionary mapping column names to their data types.
        """
        response = self.glue_client.get_table(
            DatabaseName=self.database, Name=table_name
        )
        columns = response["Table"]["StorageDescriptor"]["Columns"]
        return {col["Name"]: col["Type"] for col in columns}

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load table schemas as Document objects.

        Yields:
            Document objects, each representing the schema of a table.
        """
        table_names = self._fetch_tables()
        for table_name in table_names:
            schema = self._fetch_table_schema(table_name)
            page_content = (
                f"Database: {self.database}\nTable: {table_name}\nSchema:\n"
                + "\n".join(f"{col}: {dtype}" for col, dtype in schema.items())
            )
            doc = Document(
                page_content=page_content, metadata={"table_name": table_name}
            )
            yield doc
