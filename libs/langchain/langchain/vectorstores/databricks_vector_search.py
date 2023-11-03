import logging
import uuid
from typing import Type, List, Optional, Any, Iterable, TYPE_CHECKING, Tuple

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore, VST

if TYPE_CHECKING:
    from databricks.vector_search.client import VectorSearchClient

logger = logging.getLogger(__name__)


class DatabricksVectorSearch(VectorStore):
    """`Databricks Vector Search` vector store.

    To use, you should have the ``databricks-vectorsearch`` python package installed.

    TODO: add examples

    TODO: split this into two classes: one for delta-sync index and one for non-delta-sync index
    """

    def __init__(
        self,
        *,
        endpoint_name: str,
        index_name: str,
        text_column: Optional[str],
        columns: Optional[
            List[str]
        ] = None,  # list of column names to get when doing the search
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ):
        try:
            from databricks.vector_search.client import VectorSearchClient
        except ImportError:
            raise ImportError(
                "Could not import databricks-vectorsearch python package. "
                "Please install it with `pip install databricks-vectorsearch`."
            )
        # VS client
        self.client = VectorSearchClient(**kwargs)

        # endpoint
        self.endpoint_name = endpoint_name
        self._endpoint_health_check()

        # index
        self.index_name = index_name
        self.index = self.client.get_index(
            endpoint_name=self.endpoint_name, index_name=self.index_name
        )
        self.index_details = self.index.describe()

        # TODO: for delta-sync index with managed embedding, we can get source column from index_details
        # If customer set source_column, we should validate that it is equal to the source column in index_details
        if self._is_delta_sync_index() and self._is_managed_embedding():
            if text_column is not None:
                assert (
                    text_column == self._embedding_source_column_name()
                ), "source_column must be the same as the source column in the index."
            self.text_column = self._embedding_source_column_name()
        else:
            if text_column is None:
                raise ValueError("A source column is required for this index.")
            self.text_column = text_column

        # primary_key
        self.primary_key = self.index_details["primary_key"]

        # columns
        self.columns = columns or []
        # add primary key column and source column if not in columns
        if self.primary_key not in self.columns:
            self.columns.append(self.primary_key)
        if self.text_column not in self.columns:
            self.columns.append(self.text_column)

        # embedding model
        if not self._is_delta_sync_index() or not self._is_managed_embedding():
            if not embeddings:
                raise ValueError("An embedding function is required for this index.")
            # TODO: validate embedding dimension
            self._embedding = embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self._add(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embeddings

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        self.index.delete(primary_keys=ids)
        return True

    def similarity_search(
        self, query: str, k: int = 4, filters: Optional[Any] = None, **kwargs: Any
    ) -> List[Document]:
        docs_with_score = self._search(query=query, k=k, filters=filters, **kwargs)
        return [doc for doc, _ in docs_with_score]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filters: Optional[Any] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self._search(query=query, k=k, filters=filters, **kwargs)

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        endpoint_name: Optional[str] = None,  # required
        index_name: Optional[str] = None,  # required
        ids: Optional[List[str]] = None,
        id_column: Optional[str] = None,  # required
        dimension: Optional[int] = None,
        text_column: Optional[str] = None,
        vector_column: Optional[str] = None,
        schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> VST:
        # Validate args
        if not endpoint_name:
            raise ValueError("endpoint_name is required.")
        if not index_name:
            raise ValueError("index_name is required.")
        # TODO: check index_name 3-level format
        if not id_column:
            raise ValueError("id_column is required.")

        # Infer dimension if not provided
        if not dimension:
            dimension = len(embedding.embed_query("test query"))

        # Use default text_column and vector_column if not provided
        if not text_column:
            text_column = "text"
        if not vector_column:
            vector_column = "text_vector"

        if schema:
            if id_column not in schema:
                schema[id_column] = "string"
            if text_column not in schema:
                schema[text_column] = "string"
            if vector_column not in schema:
                schema[vector_column] = "array<float>"
        else:
            # Infer schema
            schema = {
                id_column: "string",
                text_column: "string",
                vector_column: "array<float>",
            }
            for metadata in metadatas or []:
                for key, value in metadata.items():
                    if key not in schema:
                        schema[key] = cls._infer_data_type(value)

        VectorSearchClient(**kwargs).create_direct_access_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            primary_key=id_column,
            embedding_dimension=dimension,
            embedding_vector_column=vector_column,
            schema=schema,
        )
        vs = cls(
            endpoint_name=endpoint_name,
            index_name=index_name,
            text_column=text_column,
            columns=schema.keys(),
            embeddings=embedding,
            **kwargs,
        )
        vs.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

        return vs

    @staticmethod
    def _infer_data_type(data: Any) -> str:
        """
        Supported data types:
          - integer
          - long
          - float
          - double
          - boolean
          - string
          - date
          - array<float>
        """
        if isinstance(data, int):
            return "integer"  # TODO: check if it should be long
        if isinstance(data, float):
            return "float"  # TODO: check if it should be double
        if isinstance(data, bool):
            return "boolean"
        if isinstance(data, str):
            return "string"
        if isinstance(data, list) and all(isinstance(x, float) for x in data):
            return "array<float>"
        raise ValueError(f"Unsupported data type: {type(data)}")

    def _add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if self._is_delta_sync_index():
            raise ValueError("Cannot add texts/docs to a delta sync index.")

        texts = list(texts)
        vectors = self.embeddings.embed_documents(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        updates = [
            {
                self.primary_key: id_,
                self.text_column: text,
                self._embedding_vector_column_name(): vector,
                **metadata,
            }
            for text, vector, id_, metadata in zip(texts, vectors, ids, metadatas)
        ]

        self.index.upsert(updates)

        return ids

    def _search(
        self, query: str, k: int = 4, filters: Optional[Any] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if self._is_delta_sync_index() and self._is_managed_embedding():
            query_text = query
            query_vector = None
        else:
            query_text = None
            query_vector = self.embeddings.embed_query(query)

        search_resp = self.index.similarity_search(
            columns=self.columns,
            query_text=query_text,
            query_vector=query_vector,
            filters=filters,
            num_results=k,
        )
        return self._parse_search_response(search_resp)

    def _parse_search_response(self, search_resp: dict) -> List[Tuple[Document, float]]:
        columns = [col["name"] for col in search_resp["manifest"]["columns"]]
        docs_with_score = []
        for result in search_resp["result"]["data_array"]:
            doc_id = result[columns.index(self.primary_key)]
            text_content = result[columns.index(self.text_column)]
            metadata = {
                col: value
                for col, value in zip(columns[:-1], result[:-1])
                if col not in [self.primary_key, self.text_column]
            }
            score = result[-1]
            doc = Document(id=doc_id, page_content=text_content, metadata=metadata)
            docs_with_score.append((doc, score))
        return docs_with_score

    def _embedding_source_column_name(self) -> Optional[str]:
        embedding_source_columns = self.index_details["index_spec"].get(
            "embedding_source_columns"
        )
        if (
            embedding_source_columns
            and len(embedding_source_columns) > 0
            and embedding_source_columns[0]
        ):
            return embedding_source_columns[0].get("name")

    def _embedding_vector_column_name(self) -> Optional[str]:
        embedding_vector_columns = self.index_details["index_spec"].get(
            "embedding_vector_columns"
        )
        if (
            embedding_vector_columns
            and len(embedding_vector_columns) > 0
            and embedding_vector_columns[0]
        ):
            return embedding_vector_columns[0].get("name")

    def _embedding_vector_column_dimension(self) -> Optional[int]:
        embedding_vector_columns = self.index_details["index_spec"].get(
            "embedding_vector_columns"
        )
        if (
            embedding_vector_columns
            and len(embedding_vector_columns) > 0
            and embedding_vector_columns[0]
        ):
            return embedding_vector_columns[0].get("embedding_dimension")

    def _require_embedding(self) -> None:
        """Raise an error if the embedding is not provided."""
        if self.embeddings is None:
            raise ValueError("You must provide an embedding function to search")

    def _is_delta_sync_index(self) -> bool:
        """Return whether the index is a delta sync index."""
        return self.index_details["index_type"] == "DELTA_SYNC"

    def _is_managed_embedding(self) -> bool:
        """Return whether the embedding is managed by Databricks Vector Search."""
        return self._embedding_source_column_name() is not None

    def _endpoint_health_check(self) -> None:
        """Check if the endpoint exists and is in ONLINE state."""
        resp = self.client.get_endpoint(self.endpoint_name)
        state = resp["endpoint_status"]["state"]
        if not state == "ONLINE":
            raise ValueError(f"Endpoint is not in ONLINE state. Current state: {state}")
