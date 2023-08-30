from __future__ import annotations

import json
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

if TYPE_CHECKING:
    import marqo


class Marqo(VectorStore):
    """`Marqo` vector store.

    Marqo indexes have their own models associated with them to generate your
    embeddings. This means that you can selected from a range of different models
    and also use CLIP models to create multimodal indexes
    with images and text together.

    Marqo also supports more advanced queries with multiple weighted terms, see See
    https://docs.marqo.ai/latest/#searching-using-weights-in-queries.
    This class can flexibly take strings or dictionaries for weighted queries
    in its similarity search methods.

    To use, you should have the `marqo` python package installed, you can do this with
    `pip install marqo`.

    Example:
        .. code-block:: python

            import marqo
            from langchain.vectorstores import Marqo
            client = marqo.Client(url=os.environ["MARQO_URL"], ...)
            vectorstore = Marqo(client, index_name)

    """

    def __init__(
        self,
        client: marqo.Client,
        index_name: str,
        add_documents_settings: Optional[Dict[str, Any]] = None,
        searchable_attributes: Optional[List[str]] = None,
        page_content_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
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
        self._add_documents_settings = (
            {} if add_documents_settings is None else add_documents_settings
        )
        self._searchable_attributes = searchable_attributes
        self.page_content_builder = page_content_builder

        self.tensor_fields = ["text"]

        self._document_batch_size = 1024

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload texts with metadata (properties) to Marqo.

        You can either have marqo generate ids for each document or you can provide
        your own by including a "_id" field in the metadata objects.

        Args:
            texts (Iterable[str]): am iterator of texts - assumed to preserve an
            order that matches the metadatas.
            metadatas (Optional[List[dict]], optional): a list of metadatas.

        Raises:
            ValueError: if metadatas is provided and the number of metadatas differs
            from the number of texts.

        Returns:
            List[str]: The list of ids that were added.
        """

        if self._client.index(self._index_name).get_settings()["index_defaults"][
            "treat_urls_and_pointers_as_images"
        ]:
            raise ValueError(
                "Marqo.add_texts is disabled for multimodal indexes. To add documents "
                "with a multimodal index use the Python client for Marqo directly."
            )
        documents: List[Dict[str, str]] = []

        num_docs = 0
        for i, text in enumerate(texts):
            doc = {
                "text": text,
                "metadata": json.dumps(metadatas[i]) if metadatas else json.dumps({}),
            }
            documents.append(doc)
            num_docs += 1

        ids = []
        for i in range(0, num_docs, self._document_batch_size):
            response = self._client.index(self._index_name).add_documents(
                documents[i : i + self._document_batch_size],
                tensor_fields=self.tensor_fields,
                **self._add_documents_settings,
            )
            if response["errors"]:
                err_msg = (
                    f"Error in upload for documents in index range [{i},"
                    f"{i + self._document_batch_size}], "
                    f"check Marqo logs."
                )
                raise RuntimeError(err_msg)

            ids += [item["_id"] for item in response["items"]]

        return ids

    def similarity_search(
        self,
        query: Union[str, Dict[str, float]],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search the marqo index for the most similar documents.

        Args:
            query (Union[str, Dict[str, float]]): The query for the search, either
            as a string or a weighted query.
            k (int, optional): The number of documents to return. Defaults to 4.

        Returns:
            List[Document]: k documents ordered from best to worst match.
        """
        results = self.marqo_similarity_search(query=query, k=k)

        documents = self._construct_documents_from_results_without_score(results)
        return documents

    def similarity_search_with_score(
        self,
        query: Union[str, Dict[str, float]],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Return documents from Marqo that are similar to the query as well
        as their scores.

        Args:
            query (str): The query to search with, either as a string or a weighted
            query.
            k (int, optional): The number of documents to return. Defaults to 4.

        Returns:
            List[Tuple[Document, float]]: The matching documents and their scores,
            ordered by descending score.
        """
        results = self.marqo_similarity_search(query=query, k=k)

        scored_documents = self._construct_documents_from_results_with_score(results)
        return scored_documents

    def bulk_similarity_search(
        self,
        queries: Iterable[Union[str, Dict[str, float]]],
        k: int = 4,
        **kwargs: Any,
    ) -> List[List[Document]]:
        """Search the marqo index for the most similar documents in bulk with multiple
        queries.

        Args:
            queries (Iterable[Union[str, Dict[str, float]]]): An iterable of queries to
            execute in bulk, queries in the list can be strings or dictionaries of
            weighted queries.
            k (int, optional): The number of documents to return for each query.
            Defaults to 4.

        Returns:
            List[List[Document]]: A list of results for each query.
        """
        bulk_results = self.marqo_bulk_similarity_search(queries=queries, k=k)
        bulk_documents: List[List[Document]] = []
        for results in bulk_results["result"]:
            documents = self._construct_documents_from_results_without_score(results)
            bulk_documents.append(documents)

        return bulk_documents

    def bulk_similarity_search_with_score(
        self,
        queries: Iterable[Union[str, Dict[str, float]]],
        k: int = 4,
        **kwargs: Any,
    ) -> List[List[Tuple[Document, float]]]:
        """Return documents from Marqo that are similar to the query as well as
        their scores using a batch of queries.

        Args:
            query (Iterable[Union[str, Dict[str, float]]]): An iterable of queries
            to execute in bulk, queries in the list can be strings or dictionaries
            of weighted queries.
            k (int, optional): The number of documents to return. Defaults to 4.

        Returns:
            List[Tuple[Document, float]]: A list of lists of the matching
            documents and their scores for each query
        """
        bulk_results = self.marqo_bulk_similarity_search(queries=queries, k=k)
        bulk_documents: List[List[Tuple[Document, float]]] = []
        for results in bulk_results["result"]:
            documents = self._construct_documents_from_results_with_score(results)
            bulk_documents.append(documents)

        return bulk_documents

    def _construct_documents_from_results_with_score(
        self, results: Dict[str, List[Dict[str, str]]]
    ) -> List[Tuple[Document, Any]]:
        """Helper to convert Marqo results into documents.

        Args:
            results (List[dict]): A marqo results object with the 'hits'.
            include_scores (bool, optional): Include scores alongside documents.
            Defaults to False.

        Returns:
            Union[List[Document], List[Tuple[Document, float]]]: The documents or
            document score pairs if `include_scores` is true.
        """
        documents: List[Tuple[Document, Any]] = []
        for res in results["hits"]:
            if self.page_content_builder is None:
                text = res["text"]
            else:
                text = self.page_content_builder(res)

            metadata = json.loads(res.get("metadata", "{}"))
            documents.append(
                (Document(page_content=text, metadata=metadata), res["_score"])
            )
        return documents

    def _construct_documents_from_results_without_score(
        self, results: Dict[str, List[Dict[str, str]]]
    ) -> List[Document]:
        """Helper to convert Marqo results into documents.

        Args:
            results (List[dict]): A marqo results object with the 'hits'.
            include_scores (bool, optional): Include scores alongside documents.
            Defaults to False.

        Returns:
            Union[List[Document], List[Tuple[Document, float]]]: The documents or
            document score pairs if `include_scores` is true.
        """
        documents: List[Document] = []
        for res in results["hits"]:
            if self.page_content_builder is None:
                text = res["text"]
            else:
                text = self.page_content_builder(res)

            metadata = json.loads(res.get("metadata", "{}"))
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def marqo_similarity_search(
        self,
        query: Union[str, Dict[str, float]],
        k: int = 4,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Return documents from Marqo exposing Marqo's output directly

        Args:
            query (str): The query to search with.
            k (int, optional): The number of documents to return. Defaults to 4.

        Returns:
            List[Dict[str, Any]]: This hits from marqo.
        """
        results = self._client.index(self._index_name).search(
            q=query, searchable_attributes=self._searchable_attributes, limit=k
        )
        return results

    def marqo_bulk_similarity_search(
        self, queries: Iterable[Union[str, Dict[str, float]]], k: int = 4
    ) -> Dict[str, List[Dict[str, List[Dict[str, str]]]]]:
        """Return documents from Marqo using a bulk search, exposes Marqo's
        output directly

        Args:
            queries (Iterable[Union[str, Dict[str, float]]]): A list of queries.
            k (int, optional): The number of documents to return for each query.
            Defaults to 4.

        Returns:
            Dict[str, Dict[List[Dict[str, Dict[str, Any]]]]]: A bulk search results
            object
        """
        bulk_results = {
            "result": [
                self._client.index(self._index_name).search(
                    q=query, searchable_attributes=self._searchable_attributes, limit=k
                )
                for query in queries
            ]
        }

        return bulk_results

    @classmethod
    def from_documents(
        cls: Type[Marqo],
        documents: List[Document],
        embedding: Union[Embeddings, None] = None,
        **kwargs: Any,
    ) -> Marqo:
        """Return VectorStore initialized from documents. Note that Marqo does not
        need embeddings, we retain the parameter to adhere to the Liskov substitution
        principle.


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
        cls,
        texts: List[str],
        embedding: Any = None,
        metadatas: Optional[List[dict]] = None,
        index_name: str = "",
        url: str = "http://localhost:8882",
        api_key: str = "",
        add_documents_settings: Optional[Dict[str, Any]] = None,
        searchable_attributes: Optional[List[str]] = None,
        page_content_builder: Optional[Callable[[Dict[str, str]], str]] = None,
        index_settings: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Marqo:
        """Return Marqo initialized from texts. Note that Marqo does not need
        embeddings, we retain the parameter to adhere to the Liskov
        substitution principle.

        This is a quick way to get started with marqo - simply provide your texts and
        metadatas and this will create an instance of the data store and index the
        provided data.

        To know the ids of your documents with this approach you will need to include
        them in under the key "_id" in your metadatas for each text

        Example:
        .. code-block:: python

                from langchain.vectorstores import Marqo

                datastore = Marqo(texts=['text'], index_name='my-first-index',
                url='http://localhost:8882')

        Args:
            texts (List[str]): A list of texts to index into marqo upon creation.
            embedding (Any, optional): Embeddings (not required). Defaults to None.
            index_name (str, optional): The name of the index to use, if none is
            provided then one will be created with a UUID. Defaults to None.
            url (str, optional): The URL for Marqo. Defaults to "http://localhost:8882".
            api_key (str, optional): The API key for Marqo. Defaults to "".
            metadatas (Optional[List[dict]], optional): A list of metadatas, to
            accompany the texts. Defaults to None.
            this is only used when a new index is being created. Defaults to "cpu". Can
            be "cpu" or "cuda".
            add_documents_settings (Optional[Dict[str, Any]], optional): Settings
            for adding documents, see
            https://docs.marqo.ai/0.0.16/API-Reference/documents/#query-parameters.
            Defaults to {}.
            index_settings (Optional[Dict[str, Any]], optional): Index settings if
            the index doesn't exist, see
            https://docs.marqo.ai/0.0.16/API-Reference/indexes/#index-defaults-object.
            Defaults to {}.

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

        client = marqo.Client(url=url, api_key=api_key)

        try:
            client.create_index(index_name, settings_dict=index_settings or {})
            if verbose:
                print(f"Created {index_name} successfully.")
        except Exception:
            if verbose:
                print(f"Index {index_name} exists.")

        instance: Marqo = cls(
            client,
            index_name,
            searchable_attributes=searchable_attributes,
            add_documents_settings=add_documents_settings or {},
            page_content_builder=page_content_builder,
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
