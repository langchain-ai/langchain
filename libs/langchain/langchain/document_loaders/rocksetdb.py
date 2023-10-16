from typing import Any, Callable, Iterator, List, Optional, Tuple

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


def default_joiner(docs: List[Tuple[str, Any]]) -> str:
    """Default joiner for content columns."""
    return "\n".join([doc[1] for doc in docs])


class ColumnNotFoundError(Exception):
    """Column not found error."""

    def __init__(self, missing_key: str, query: str):
        super().__init__(f'Column "{missing_key}" not selected in query:\n{query}')


class RocksetLoader(BaseLoader):
    """Load from a `Rockset` database.

    To use, you should have the `rockset` python package installed.

    Example:
        .. code-block:: python

            # This code will load 3 records from the "langchain_demo"
            # collection as Documents, with the `text` column used as
            # the content

            from langchain.document_loaders import RocksetLoader
            from rockset import RocksetClient, Regions, models

            loader = RocksetLoader(
                RocksetClient(Regions.usw2a1, "<api key>"),
                models.QueryRequestSql(
                    query="select * from langchain_demo limit 3"
                ),
                ["text"]
            )
        )
    """

    def __init__(
        self,
        client: Any,
        query: Any,
        content_keys: List[str],
        metadata_keys: Optional[List[str]] = None,
        content_columns_joiner: Callable[[List[Tuple[str, Any]]], str] = default_joiner,
    ):
        """Initialize with Rockset client.

        Args:
            client: Rockset client object.
            query: Rockset query object.
            content_keys: The collection columns to be written into the `page_content`
                of the Documents.
            metadata_keys: The collection columns to be written into the `metadata` of
                the Documents. By default, this is all the keys in the document.
            content_columns_joiner: Method that joins content_keys and its values into a
                string. It's method that takes in a List[Tuple[str, Any]]],
                representing a list of tuples of (column name, column value).
                By default, this is a method that joins each column value with a new
                line. This method is only relevant if there are multiple content_keys.
        """
        try:
            from rockset import QueryPaginator, RocksetClient
            from rockset.models import QueryRequestSql
        except ImportError:
            raise ImportError(
                "Could not import rockset client python package. "
                "Please install it with `pip install rockset`."
            )

        if not isinstance(client, RocksetClient):
            raise ValueError(
                f"client should be an instance of rockset.RocksetClient, "
                f"got {type(client)}"
            )

        if not isinstance(query, QueryRequestSql):
            raise ValueError(
                f"query should be an instance of rockset.model.QueryRequestSql, "
                f"got {type(query)}"
            )

        self.client = client
        self.query = query
        self.content_keys = content_keys
        self.content_columns_joiner = content_columns_joiner
        self.metadata_keys = metadata_keys
        self.paginator = QueryPaginator
        self.request_model = QueryRequestSql

        try:
            self.client.set_application("langchain")
        except AttributeError:
            # ignore
            pass

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        query_results = self.client.Queries.query(
            sql=self.query
        ).results  # execute the SQL query
        for doc in query_results:  # for each doc in the response
            try:
                yield Document(
                    page_content=self.content_columns_joiner(
                        [(col, doc[col]) for col in self.content_keys]
                    ),
                    metadata={col: doc[col] for col in self.metadata_keys}
                    if self.metadata_keys is not None
                    else doc,
                )  # try to yield the Document
            except (
                KeyError
            ) as e:  # either content_columns or metadata_columns is invalid
                raise ColumnNotFoundError(
                    e.args[0], self.query
                )  # raise that the column isn't in the db schema
