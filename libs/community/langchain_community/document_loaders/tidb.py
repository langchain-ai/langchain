from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class TiDBLoader(BaseLoader):
    """Load documents from TiDB."""

    def __init__(
        self,
        connection_string: str,
        query: str,
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        engine_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize TiDB document loader.

        Args:
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@127.0.0.1:4000/test".
            query: The query to run in TiDB.
            page_content_columns: Optional. Columns written to Document `page_content`,
                default(None) to all columns.
            metadata_columns: Optional. Columns written to Document `metadata`,
                default(None) to no columns.
            engine_args: Optional. Additional arguments to pass to sqlalchemy engine.
        """
        self.connection_string = connection_string
        self.query = query
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns if metadata_columns is not None else []
        self.engine_args = engine_args

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load TiDB data into document objects."""

        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        from sqlalchemy.sql import text

        # use sqlalchemy to create db connection
        engine: Engine = create_engine(
            self.connection_string, **(self.engine_args or {})
        )

        # execute query
        with engine.connect() as conn:
            result = conn.execute(text(self.query))

            # convert result to Document objects
            column_names = list(result.keys())
            for row in result:
                # convert row to dict{column:value}
                row_data = {
                    column_names[index]: value for index, value in enumerate(row)
                }
                page_content = "\n".join(
                    f"{k}: {v}"
                    for k, v in row_data.items()
                    if self.page_content_columns is None
                    or k in self.page_content_columns
                )
                metadata = {col: row_data[col] for col in self.metadata_columns}
                yield Document(page_content=page_content, metadata=metadata)

    def load(self) -> List[Document]:
        """Load TiDB data into document objects."""
        return list(self.lazy_load())
