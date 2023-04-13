"""Wrapper around weaviate vector database."""
from __future__ import annotations

import json
import uuid
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore

if TYPE_CHECKING:
    import marqo


class Marqo(VectorStore):
    """Wrapper around Marqo database.

    Marqo indexes have their own models associated with them to generate your embeddings. This means that
    you can selected from a range of different models and also use CLIP models to create multimodal indexes
    with images and text together.

    Marqo also supports more advanced queries with mutliple weighted terms, see See https://docs.marqo.ai/latest/#searching-using-weights-in-queries.
    This class can flexibly take strings or dictionaries for weighted queries in its similarity search methods.

    To use, you should have the `marqo` python package installed, you can do this with `pip install marqo`.

    Example:
        .. code-block:: python

            import marqo
            from langchain.vectorstores import Marqo
            client = marqo.Client(url=os.environ["MARQO_URL"], ...)
            datastore = Marqo(client, index_name)

    """

    def __init__(
        self,
        client: marqo.Client,
        index_name: str,
        add_documents_settings: Optional[Dict[str, Any]] = {},
        searchable_attributes: Optional[List[str]] = None,
    ):
        """Initialize with Marqo client."""
        try:
            import marqo
        except ImportError:
            raise ValueError(
                "Could not import marqo python package. "
                "Please install it with `pip install marqo`."
            )
        if not isinstance(client, marqo.Client):
            raise ValueError(
                f"client should be an instance of marqo.Client, got {type(client)}"
            )
        self._client = client
        self._index_name = index_name
        self._add_documents_settings = add_documents_settings

        self._non_tensor_fields = ["metadata"]
        self._searchable_attributes = searchable_attributes
        self._document_batch_size = 1024

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Upload texts with metadata (properties) to Marqo.

        You can either have marqo generate ids for each document or you can provide your own by including
        a "_id" field in the metadata objects.

        Args:
            texts (Iterable[str]): am iterator of texts - assumed to preserve an order that matches the metadatas.
            metadatas (Optional[List[dict]], optional): a list of metadatas.

        Raises:
            ValueError: if metadatas is provided and the number of metadatas differs from the number of texts.

        Returns:
            List[str]: The list of ids that were added.
        """

        if self._client.index(self._index_name).get_settings()["index_defaults"][
            "treat_urls_and_pointers_as_images"
        ]:
            raise ValueError(
                "Marqo.add_texts is disabled for multimodal indexes. To add documents use the Python Marqo client."
            )

        if metadatas and len(texts) != len(metadatas):
            raise ValueError(
                f"The lengths of texts and metadatas must be the same, {len(texts)} texts were provided and {len(metadatas)} metadatas were provided."
            )

        documents: List[Dict[str, Union[str, int, float, List[str]]]] = []

        for i, text in enumerate(texts):
            doc = {"text": text}
            doc["metadata"] = json.dumps(metadatas[i]) if metadatas else json.dumps({})
            documents.append(doc)

        ids = []
        for i in range(0, len(documents), self._document_batch_size):
            response = self._client.index(self._index_name).add_documents(
                documents[i : i + self._document_batch_size],
                non_tensor_fields=self._non_tensor_fields,
                **self._add_documents_settings,
            )
            if response["errors"]:
                warnings.warn(
                    f"Error in upload for documents in index range [{i},{i+self._document_batch_size}], check Marqo logs."
                )
            ids += [item["_id"] for item in response["items"]]

        return ids

    def similarity_search(
        self,
        query: Union[str, Dict[str, float]],
        k: int = 4,
        page_content_builder: Optional[Callable[[dict], str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search the marqo index for the most similar documents.

        Args:
            query (Union[str, Dict[str, float]]): The query for the search, either as a string or a weighted query.
            k (int, optional): The number of documents to return. Defaults to 4.
            page_content_builder (Optional[Callable[[dict], str]], optional): A callback to format arbitrary documents into a string. Defaults to None.

        Returns:
            List[Document]: k documents ordered from best to worst match.
        """
        results = self._client.index(self._index_name).search(
            q=query, searchable_attributes=self._searchable_attributes, limit=k
        )
        documents = self._construct_documents_from_results(
            results, page_content_builder=page_content_builder, include_scores=False
        )
        return documents

    def bulk_similarity_search(
        self,
        queries: Iterable[Union[str, Dict[str, float]]],
        k: int = 4,
        page_content_builder: Optional[Callable[[dict], str]] = None,
        **kwargs: Any,
    ) -> List[List[Document]]:
        """Search the marqo index for the most similar documents in bulk with multiple queries.

        Args:
            queries (Iterable[Union[str, Dict[str, float]]]): An iterable of queries to execute in bulk, queries in the list can be strings or dictonaries of weighted queries.
            k (int, optional): The number of documents to return for each query. Defaults to 4.
            page_content_builder (Optional[Callable[[dict], str]], optional): A callback to format arbitrary documents into a string. Defaults to None.

        Returns:
            List[List[Document]]: A list of results for each query.
        """
        bulk_results = self._client.bulk_search(
            [
                {
                    "index": self._index_name,
                    "q": query,
                    "searchable_attributes": self._searchable_attributes,
                    "limit": k,
                }
                for query in queries
            ]
        )
        bulk_documents: List[List[Document]] = []
        for results in bulk_results["result"]:
            documents = self._construct_documents_from_results(
                results, page_content_builder=page_content_builder, include_scores=False
            )
            bulk_documents.append(documents)

        return bulk_documents

    def similarity_search_with_score(
        self,
        query: Union[str, Dict[str, float]],
        k: int = 4,
        page_content_builder: Optional[Callable[[dict], str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return documents from Marqo that are similar to the query as well as their scores.

        Args:
            query (str): The query to search with, either as a string or a weighted query.
            k (int, optional): The number of documents to return. Defaults to 4.
            page_content_builder (Optional[Callable[[dict], str]], optional): A callback to format arbitrary documents into a string. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: The matching documents and their scores, ordered by descending score.
        """
        results = self._client.index(self._index_name).search(q=query, limit=k)
        scored_documents = self._construct_documents_from_results(
            results, page_content_builder=page_content_builder, include_scores=True
        )
        return scored_documents

    def bulk_similarity_search_with_score(
        self,
        queries: Iterable[Union[str, Dict[str, float]]],
        k: int = 4,
        page_content_builder: Optional[Callable[[dict], str]] = None,
        **kwargs: Any,
    ) -> List[List[Tuple[Document, float]]]:
        """Return documents from Marqo that are similar to the query as well as their scores using
        a batch of queries.

        Args:
            query (Iterable[Union[str, Dict[str, float]]]): An iterable of queries to execute in bulk, queries in the list can be strings or dictonaries of weighted queries.
            k (int, optional): The number of documents to return. Defaults to 4.
            page_content_builder (Optional[Callable[[dict], str]], optional): A callback to format arbitrary documents into a string. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: A list of lists of the matching documents and their scores for each query
        """
        bulk_results = self._client.bulk_search(
            [
                {
                    "index": self._index_name,
                    "q": query,
                    "searchable_attributes": self._searchable_attributes,
                    "limit": k,
                }
                for query in queries
            ]
        )
        bulk_documents: List[List[Tuple[Document, float]]] = []
        for results in bulk_results["result"]:
            documents = self._construct_documents_from_results(
                results, page_content_builder=page_content_builder, include_scores=True
            )
            bulk_documents.append(documents)

        return bulk_documents

    def _construct_documents_from_results(
        self,
        results: List[dict],
        page_content_builder: Optional[Callable[[dict], str]] = None,
        include_scores: bool = False,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Helper to convert Marqo results into documents.

        Args:
            results (List[dict]): A marqo results object with the 'hits'.
            include_scores (bool, optional): Include scores alongside documents. Defaults to False.
            page_content_builder (Optional[Callable[[dict], str]], optional): A callback to format arbitrary documents into a string. Defaults to None.

        Returns:
            Union[List[Document], List[Tuple[Document, float]]]: The documents or document score pairs if `include_scores` is true.
        """
        documents: Union[List[Document], List[Tuple[Document, float]]] = []
        for res in results["hits"]:
            if page_content_builder is None:
                text = res["text"]
            else:
                text = page_content_builder(res)

            md = res.get("metadata")
            metadata = json.loads(md if md else "{}")
            if include_scores:
                documents.append(
                    (Document(page_content=text, metadata=metadata), res["_score"])
                )
            else:
                documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def marqo_similarity_search(
        self,
        query: Union[str, Dict[str, float]],
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Return documents from Marqo exposing Marqo's output directly

        Args:
            query (str): The query to search with.
            k (int, optional): The number of documents to return. Defaults to 4.

        Returns:
            List[Dict[str, Any]]: This hits from marqo.
        """
        results = self._client.index(self._index_name).search(q=query, limit=k)
        return results["hits"]

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Any = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Return VectorStore initialized from documents. Note that Marqo does not need embeddings, we retain the
        parameter to adhere to the Liskov substitution principle.


        Args:
            documents (List[Document]): Input documents
            embedding (Any, optional): Embeddings (not required). Defaults to None.

        Returns:
            VectorStore: A Marqo vectorstore
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, metadatas=metadatas, **kwargs)

    @classmethod
    def from_texts(
        cls: Marqo,
        texts: List[str],
        embedding: Any = None,
        metadatas: Optional[List[dict]] = None,
        index_name: str = None,
        url: str = "http://localhost:8882",
        api_key: str = "",
        marqo_device: str = "cpu",
        add_documents_settings: Optional[Dict[str, Any]] = {},
        searchable_attributes: Optional[List[str]] = None,
        index_settings: Optional[Dict[str, Any]] = {},
        verbose: bool = True,
        **kwargs: Any,
    ) -> Marqo:
        """Return Marqo initialized from texts. Note that Marqo does not need embeddings, we retain the
        parameter to adhere to the Liskov substitution principle.

        This is a quick way to get started with marqo - simply provide your texts and metadatas and this
        will create an instance of the data store and index the provided data.

        To know the ids of your documents with this approach you will need to include them in under the key "_id"
        in your metadatas for each text

        Example:
        .. code-block:: python

                from langchain.vectorstores import Marqo

                datastore = Marqo(texts=['text'], index_name='my-first-index', url='http://localhost:8882')

        Args:
            texts (List[str]): A list of texts to index into marqo upon creation.
            embedding (Any, optional): Embeddings (not required). Defaults to None.
            index_name (str, optional): The name of the index to use, if none is provided then one will be created with a UUID. Defaults to None.
            url (str, optional): The URL for Marqo. Defaults to "http://localhost:8882".
            api_key (str, optional): The API key for Marqo. Defaults to "".
            metadatas (Optional[List[dict]], optional): A list of metadatas, to accompany the texts. Defaults to None.
            marqo_device (str, optional): The device for the marqo to use on the server, this is only used when a new index is being created. Defaults to "cpu".
            add_documents_settings (Optional[Dict[str, Any]], optional): Settings for adding documents, see https://docs.marqo.ai/0.0.16/API-Reference/documents/#query-parameters. Defaults to {}.
            index_settings (Optional[Dict[str, Any]], optional): Index settings if the index doesn't exist, see https://docs.marqo.ai/0.0.16/API-Reference/indexes/#index-defaults-object. Defaults to {}.

        Returns:
            Marqo: An instance of the Marqo vector store
        """
        try:
            import marqo
        except ImportError:
            raise ValueError(
                "Could not import marqo python package. "
                "Please install it with `pip install marqo`."
            )

        if not index_name:
            index_name = str(uuid.uuid4())

        client = marqo.Client(url=url, api_key=api_key, indexing_device=marqo_device)

        try:
            client.create_index(index_name, settings_dict=index_settings)
            if verbose:
                print(f"Created {index_name} successfully.")
        except:
            if verbose:
                print(f"Index {index_name} exists.")

        instance: Marqo = cls(
            client,
            index_name,
            searchable_attributes=searchable_attributes,
            add_documents_settings=add_documents_settings,
        )
        instance.add_texts(texts, metadatas)
        return instance

    def get_indexes(self) -> List[Dict[str, str]]:
        """Helper to see your available indexes in marqo, useful if the
        from_texts method was used without an index name specified

        Returns:
            List[Dict[str, str]]: The list of indexes
        """
        return self._client.get_indexes()["results"]

    def get_number_of_documents(self) -> int:
        """Helper to see the number of documents in the index

        Returns:
            int: The number of documents
        """
        return self._client.index(self._index_name).get_stats()["numberOfDocuments"]
