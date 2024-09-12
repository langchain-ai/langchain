from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from sqlalchemy.engine import RowMapping
from sqlalchemy.sql.expression import Select

from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.sql_database import SQLDatabase


class SQLDatabaseLoader(BaseLoader):
    """
    Load documents by querying database tables supported by SQLAlchemy.

    For talking to the database, the document loader uses the `SQLDatabase`
    utility from the LangChain integration toolkit.

    Each document represents one row of the result.
    """

    def __init__(
        self,
        query: Union[str, Select],
        db: SQLDatabase,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        page_content_mapper: Optional[Callable[..., str]] = None,
        metadata_mapper: Optional[Callable[..., Dict[str, Any]]] = None,
        source_columns: Optional[Sequence[str]] = None,
        include_rownum_into_metadata: bool = False,
        include_query_into_metadata: bool = False,
    ):
        """
        Args:
            query: The query to execute.
            db: A LangChain `SQLDatabase`, wrapping an SQLAlchemy engine.
            sqlalchemy_kwargs: More keyword arguments for SQLAlchemy's `create_engine`.
            parameters: Optional. Parameters to pass to the query.
            page_content_mapper: Optional. Function to convert a row into a string
              to use as the `page_content` of the document. By default, the loader
              serializes the whole row into a string, including all columns.
            metadata_mapper: Optional. Function to convert a row into a dictionary
              to use as the `metadata` of the document. By default, no columns are
              selected into the metadata dictionary.
            source_columns: Optional. The names of the columns to use as the `source`
              within the metadata dictionary.
            include_rownum_into_metadata: Optional. Whether to include the row number
              into the metadata dictionary. Default: False.
            include_query_into_metadata: Optional. Whether to include the query
              expression into the metadata dictionary. Default: False.
        """
        self.query = query
        self.db: SQLDatabase = db
        self.parameters = parameters or {}
        self.page_content_mapper = (
            page_content_mapper or self.page_content_default_mapper
        )
        self.metadata_mapper = metadata_mapper or self.metadata_default_mapper
        self.source_columns = source_columns
        self.include_rownum_into_metadata = include_rownum_into_metadata
        self.include_query_into_metadata = include_query_into_metadata

    def lazy_load(self) -> Iterator[Document]:
        try:
            import sqlalchemy as sa
        except ImportError:
            raise ImportError(
                "Could not import sqlalchemy python package. "
                "Please install it with `pip install sqlalchemy`."
            )

        # Querying in `cursor` fetch mode will return an SQLAlchemy `Result` instance.
        result: sa.Result[Any]

        # Invoke the database query.
        if isinstance(self.query, sa.SelectBase):
            result = self.db._execute(  # type: ignore[assignment]
                self.query, fetch="cursor", parameters=self.parameters
            )
            query_sql = str(self.query.compile(bind=self.db._engine))
        elif isinstance(self.query, str):
            result = self.db._execute(  # type: ignore[assignment]
                sa.text(self.query), fetch="cursor", parameters=self.parameters
            )
            query_sql = self.query
        else:
            raise TypeError(f"Unable to process query of unknown type: {self.query}")

        # Iterate database result rows and generate list of documents.
        for i, row in enumerate(result.mappings()):
            page_content = self.page_content_mapper(row)
            metadata = self.metadata_mapper(row)

            if self.include_rownum_into_metadata:
                metadata["row"] = i
            if self.include_query_into_metadata:
                metadata["query"] = query_sql

            source_values = []
            for column, value in row.items():
                if self.source_columns and column in self.source_columns:
                    source_values.append(value)
            if source_values:
                metadata["source"] = ",".join(source_values)

            yield Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def page_content_default_mapper(
        row: RowMapping, column_names: Optional[List[str]] = None
    ) -> str:
        """
        A reasonable default function to convert a record into a "page content" string.
        """
        if column_names is None:
            column_names = list(row.keys())
        return "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
            if column in column_names
        )

    @staticmethod
    def metadata_default_mapper(
        row: RowMapping, column_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        A reasonable default function to convert a record into a "metadata" dictionary.
        """
        if column_names is None:
            return {}

        metadata: Dict[str, Any] = {}
        for column, value in row.items():
            if column in column_names:
                metadata[column] = value
        return metadata
