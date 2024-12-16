"""Wrapper around Epsilla vector database."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from pyepsilla import vectordb

logger = logging.getLogger()


class Epsilla(VectorStore):
    """
    Wrapper around Epsilla vector database.

    As a prerequisite, you need to install ``pyepsilla`` package
    and have a running Epsilla vector database (for example, through our docker image)
    See the following documentation for how to run an Epsilla vector database:
    https://epsilla-inc.gitbook.io/epsilladb/quick-start

    Args:
        client (Any): Epsilla client to connect to.
        embeddings (Embeddings): Function used to embed the texts.
        db_path (Optional[str]): The path where the database will be persisted.
                                 Defaults to "/tmp/langchain-epsilla".
        db_name (Optional[str]): Give a name to the loaded database.
                                 Defaults to "langchain_store".
    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Epsilla
            from pyepsilla import vectordb

            client = vectordb.Client()
            embeddings = OpenAIEmbeddings()
            db_path = "/tmp/vectorstore"
            db_name = "langchain_store"
            epsilla = Epsilla(client, embeddings, db_path, db_name)
    """

    _LANGCHAIN_DEFAULT_DB_NAME: str = "langchain_store"
    _LANGCHAIN_DEFAULT_DB_PATH: str = "/tmp/langchain-epsilla"
    _LANGCHAIN_DEFAULT_TABLE_NAME: str = "langchain_collection"

    def __init__(
        self,
        client: Any,
        embeddings: Embeddings,
        db_path: Optional[str] = _LANGCHAIN_DEFAULT_DB_PATH,
        db_name: Optional[str] = _LANGCHAIN_DEFAULT_DB_NAME,
    ):
        """Initialize with necessary components."""
        try:
            import pyepsilla
        except ImportError as e:
            raise ImportError(
                "Could not import pyepsilla python package. "
                "Please install pyepsilla package with `pip install pyepsilla`."
            ) from e

        if not isinstance(
            client, (pyepsilla.vectordb.Client, pyepsilla.cloud.client.Vectordb)
        ):
            raise TypeError(
                "client should be an instance of pyepsilla.vectordb.Client or "
                f"pyepsilla.cloud.client.Vectordb, got {type(client)}"
            )

        self._client: vectordb.Client = client
        self._db_name = db_name
        self._embeddings = embeddings
        self._collection_name = Epsilla._LANGCHAIN_DEFAULT_TABLE_NAME
        self._client.load_db(db_name=db_name, db_path=db_path)
        self._client.use_db(db_name=db_name)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embeddings

    def use_collection(self, collection_name: str) -> None:
        """
        Set default collection to use.

        Args:
            collection_name (str): The name of the collection.
        """
        self._collection_name = collection_name

    def clear_data(self, collection_name: str = "") -> None:
        """
        Clear data in a collection.

        Args:
            collection_name (Optional[str]): The name of the collection.
                If not provided, the default collection will be used.
        """
        if not collection_name:
            collection_name = self._collection_name
        self._client.drop_table(collection_name)

    def get(
        self, collection_name: str = "", response_fields: Optional[List[str]] = None
    ) -> List[dict]:
        """Get the collection.

        Args:
            collection_name (Optional[str]): The name of the collection
                to retrieve data from.
                If not provided, the default collection will be used.
            response_fields (Optional[List[str]]): List of field names in the result.
                If not specified, all available fields will be responded.

        Returns:
            A list of the retrieved data.
        """
        if not collection_name:
            collection_name = self._collection_name
        status_code, response = self._client.get(
            table_name=collection_name, response_fields=response_fields
        )
        if status_code != 200:
            logger.error(f"Failed to get records: {response['message']}")
            raise Exception("Error: {}.".format(response["message"]))
        return response["result"]

    def _create_collection(
        self, table_name: str, embeddings: list, metadatas: Optional[list[dict]] = None
    ) -> None:
        if not embeddings:
            raise ValueError("Embeddings list is empty.")

        dim = len(embeddings[0])
        fields: List[dict] = [
            {"name": "id", "dataType": "INT"},
            {"name": "text", "dataType": "STRING"},
            {"name": "embeddings", "dataType": "VECTOR_FLOAT", "dimensions": dim},
        ]
        if metadatas is not None:
            field_names = [field["name"] for field in fields]
            for metadata in metadatas:
                for key, value in metadata.items():
                    if key in field_names:
                        continue
                    d_type: str
                    if isinstance(value, str):
                        d_type = "STRING"
                    elif isinstance(value, int):
                        d_type = "INT"
                    elif isinstance(value, float):
                        d_type = "FLOAT"
                    elif isinstance(value, bool):
                        d_type = "BOOL"
                    else:
                        raise ValueError(f"Unsupported data type for {key}.")
                    fields.append({"name": key, "dataType": d_type})
                    field_names.append(key)

        status_code, response = self._client.create_table(
            table_name, table_fields=fields
        )
        if status_code != 200:
            if status_code == 409:
                logger.info(f"Continuing with the existing table {table_name}.")
            else:
                logger.error(
                    f"Failed to create collection {table_name}: {response['message']}"
                )
                raise Exception("Error: {}.".format(response["message"]))

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        collection_name: Optional[str] = "",
        drop_old: Optional[bool] = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        Embed texts and add them to the database.

        Args:
            texts (Iterable[str]): The texts to embed.
            metadatas (Optional[List[dict]]): Metadata dicts
                        attached to each of the texts. Defaults to None.
            collection_name (Optional[str]): Which collection to use.
                        Defaults to "langchain_collection".
                        If provided, default collection name will be set as well.
            drop_old (Optional[bool]): Whether to drop the previous collection
                        and create a new one. Defaults to False.

        Returns:
            List of ids of the added texts.
        """
        if not collection_name:
            collection_name = self._collection_name
        else:
            self._collection_name = collection_name

        if drop_old:
            self._client.drop_db(db_name=collection_name)

        texts = list(texts)
        try:
            embeddings = self._embeddings.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self._embeddings.embed_query(x) for x in texts]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        self._create_collection(
            table_name=collection_name, embeddings=embeddings, metadatas=metadatas
        )

        ids = [hash(uuid.uuid4()) for _ in texts]
        records = []
        for index, id in enumerate(ids):
            record = {
                "id": id,
                "text": texts[index],
                "embeddings": embeddings[index],
            }
            if metadatas is not None:
                metadata = metadatas[index].items()
                for key, value in metadata:
                    record[key] = value
            records.append(record)

        status_code, response = self._client.insert(
            table_name=collection_name, records=records
        )
        if status_code != 200:
            logger.error(
                f"Failed to add records to {collection_name}: {response['message']}"
            )
            raise Exception("Error: {}.".format(response["message"]))
        return [str(id) for id in ids]

    def similarity_search(
        self, query: str, k: int = 4, collection_name: str = "", **kwargs: Any
    ) -> List[Document]:
        """
        Return the documents that are semantically most relevant to the query.

        Args:
            query (str): String to query the vectorstore with.
            k (Optional[int]): Number of documents to return. Defaults to 4.
            collection_name (Optional[str]): Collection to use.
                Defaults to "langchain_store" or the one provided before.
        Returns:
            List of documents that are semantically most relevant to the query
        """
        if not collection_name:
            collection_name = self._collection_name
        query_vector = self._embeddings.embed_query(query)
        status_code, response = self._client.query(
            table_name=collection_name,
            query_field="embeddings",
            query_vector=query_vector,
            limit=k,
        )
        if status_code != 200:
            logger.error(f"Search failed: {response['message']}.")
            raise Exception("Error: {}.".format(response["message"]))

        exclude_keys = ["id", "text", "embeddings"]
        return list(
            map(
                lambda item: Document(
                    page_content=item["text"],
                    metadata={
                        key: item[key] for key in item if key not in exclude_keys
                    },
                ),
                response["result"],
            )
        )

    @classmethod
    def from_texts(
        cls: Type[Epsilla],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Any = None,
        db_path: Optional[str] = _LANGCHAIN_DEFAULT_DB_PATH,
        db_name: Optional[str] = _LANGCHAIN_DEFAULT_DB_NAME,
        collection_name: Optional[str] = _LANGCHAIN_DEFAULT_TABLE_NAME,
        drop_old: Optional[bool] = False,
        **kwargs: Any,
    ) -> Epsilla:
        """Create an Epsilla vectorstore from raw documents.

        Args:
            texts (List[str]): List of text data to be inserted.
            embeddings (Embeddings): Embedding function.
            client (pyepsilla.vectordb.Client): Epsilla client to connect to.
            metadatas (Optional[List[dict]]): Metadata for each text.
                    Defaults to None.
            db_path (Optional[str]): The path where the database will be persisted.
                    Defaults to "/tmp/langchain-epsilla".
            db_name (Optional[str]): Give a name to the loaded database.
                    Defaults to "langchain_store".
            collection_name (Optional[str]): Which collection to use.
                    Defaults to "langchain_collection".
                    If provided, default collection name will be set as well.
            drop_old (Optional[bool]): Whether to drop the previous collection
                    and create a new one. Defaults to False.

        Returns:
            Epsilla: Epsilla vector store.
        """
        instance = Epsilla(client, embedding, db_path=db_path, db_name=db_name)
        instance.add_texts(
            texts,
            metadatas=metadatas,
            collection_name=collection_name,
            drop_old=drop_old,
            **kwargs,
        )

        return instance

    @classmethod
    def from_documents(
        cls: Type[Epsilla],
        documents: List[Document],
        embedding: Embeddings,
        client: Any = None,
        db_path: Optional[str] = _LANGCHAIN_DEFAULT_DB_PATH,
        db_name: Optional[str] = _LANGCHAIN_DEFAULT_DB_NAME,
        collection_name: Optional[str] = _LANGCHAIN_DEFAULT_TABLE_NAME,
        drop_old: Optional[bool] = False,
        **kwargs: Any,
    ) -> Epsilla:
        """Create an Epsilla vectorstore from a list of documents.

        Args:
            texts (List[str]): List of text data to be inserted.
            embeddings (Embeddings): Embedding function.
            client (pyepsilla.vectordb.Client): Epsilla client to connect to.
            metadatas (Optional[List[dict]]): Metadata for each text.
                    Defaults to None.
            db_path (Optional[str]): The path where the database will be persisted.
                    Defaults to "/tmp/langchain-epsilla".
            db_name (Optional[str]): Give a name to the loaded database.
                    Defaults to "langchain_store".
            collection_name (Optional[str]): Which collection to use.
                    Defaults to "langchain_collection".
                    If provided, default collection name will be set as well.
            drop_old (Optional[bool]): Whether to drop the previous collection
                    and create a new one. Defaults to False.

        Returns:
            Epsilla: Epsilla vector store.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            client=client,
            db_path=db_path,
            db_name=db_name,
            collection_name=collection_name,
            drop_old=drop_old,
            **kwargs,
        )
