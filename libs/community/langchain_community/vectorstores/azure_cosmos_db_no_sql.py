import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from azure.cosmos.cosmos_client import CosmosClient


# You can read more about vector search using AzureCosmosDBNoSQL here.
# https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
class AzureCosmosDBNoSqlVectorSearch(VectorStore):
    """`Azure Cosmos DB for NoSQL` vector store.

    To use, you should have both:
    - the ``azure-cosmos`` python package installed
    """

    def __init__(
        self,
        *,
        embedding: Embeddings,
        cosmos_client: Optional[CosmosClient] = None,
        database_name: Optional[str] = None,
        container_name: Optional[str] = None,
        partition_key: Optional[str] = None,
        vector_embedding_policy: Optional[Dict[str, Any]] = None,
        indexing_policy: Optional[Dict[str, Any]] = None,
        cosmos_container_properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Constructor for AzureCosmosDBNoSqlVectorSearch

        Args:
            cosmos_client: Client used to connect to azure cosmosdb no sql account.
            database_name: Name of the database to be created.
            container_name: Name of the container to be created.
            partition_key: Partition Key for the container.
            embedding: Text embedding model to use.
            vector_embedding_policy: Vector Embedding Policy for the container.
            indexing_policy: Indexing Policy for the container.
            cosmos_container_properties: Container Properties for the container.
        """
        if (
            indexing_policy["vectorIndexes"] is None
            or len(indexing_policy["vectorIndexes"]) == 0
        ):
            raise ValueError(
                "vectorIndexes cannot be null or empty in the indexing_policy."
            )
        if (
            vector_embedding_policy is None
            or len(vector_embedding_policy["vectorEmbeddings"]) == 0
        ):
            raise ValueError(
                "vectorEmbeddings cannot be null "
                "or empty in the vector_embedding_policy."
            )

        self._cosmos_client = cosmos_client
        self._database_name = database_name
        self._container_name = container_name
        self._embedding = embedding
        self._partition_key = partition_key
        self._vector_embedding_policy = vector_embedding_policy
        self._indexing_policy = indexing_policy
        self._cosmos_container_properties = cosmos_container_properties

        # Create the database if it already doesn't exist
        self._database = self._cosmos_client.create_database_if_not_exists(
            id=self._database_name
        )

        # Create the collection if it already doesn't exist
        self._container = self._database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=self._cosmos_container_properties["partition_key"],
            indexing_policy=self._indexing_policy,
            vector_embedding_policy=self._vector_embedding_policy,
        )
        self._embedding_key = self._vector_embedding_policy["vectorEmbeddings"][0][
            "path"
        ][1:]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        _metadatas = list(metadatas if metadatas is not None else ({} for _ in texts))

        return self._insert_texts(list(texts), _metadatas)

    def _insert_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Used to Load Documents into the collection

        Args:
            texts: The list of documents strings to load
            metadatas: The list of metadata objects associated with each document

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # If the texts is empty, throw an error
        if not texts:
            raise Exception("Texts can not be null or empty")

        # Embed and create the documents
        embeddings = self._embedding.embed_documents(texts)
        text_key = "text"

        to_insert = [
            {"id": str(uuid.uuid4()), text_key: t, self._embedding_key: embedding, **m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in CosmosDB No Sql
        doc_ids: List[str] = []
        for item in to_insert:
            created_doc = self._container.create_item(item)
            doc_ids.append(created_doc["id"])
        return doc_ids

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        cosmos_client: CosmosClient = None,
        database_name: Optional[str] = None,
        container_name: Optional[str] = None,
        partition_key: Optional[str] = None,
        vector_embedding_policy: Optional[Dict[str, Any]] = None,
        indexing_policy: Optional[Dict[str, Any]] = None,
        cosmos_container_properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VST:
        if cosmos_client is None:
            raise ValueError("Must provide 'cosmos_client' named parameter.")
        if database_name is None:
            raise ValueError("Must provide 'database_name' named parameter.")
        if container_name is None:
            raise ValueError("Must provide 'container_name' named parameter.")
        if partition_key is None:
            raise ValueError("Must provide 'partition_key' named parameter.")
        if vector_embedding_policy is None:
            raise ValueError("Must provide 'vector_embedding_policy' named parameter.")
        if indexing_policy is None:
            raise ValueError("Must provide 'indexing_policy' named parameter.")
        if cosmos_container_properties is None:
            raise ValueError(
                "Must provide 'cosmos_container_properties' named parameter."
            )

        vectorstore = cls(
            cosmos_client,
            database_name,
            container_name,
            partition_key,
            embedding,
            vector_embedding_policy,
            indexing_policy,
            cosmos_container_properties,
        )
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            self._container.delete_item(document_id)
        return True

    def delete_document_by_id(self, document_id: Optional[str] = None) -> None:
        """Removes a Specific Document by id

        Args:
            document_id: The document identifier
        """
        if document_id is None:
            raise ValueError("No document ids provided to delete.")
        self._container.delete_item(document_id, partition_key=document_id)

    def _similarity_search_with_score(
        self,
        embeddings: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        query = (
            "SELECT TOP {} c.id, c.{}, c.text, VectorDistance(c.{}, {}) AS "
            "SimilarityScore FROM c ORDER BY VectorDistance(c.{}, {})".format(
                k,
                self._embedding_key,
                self._embedding_key,
                embeddings,
                self._embedding_key,
                embeddings,
            )
        )
        docs_and_scores = []
        items = list(
            self._container.query_items(query=query, enable_cross_partition_query=True)
        )
        for item in items:
            text = item["text"]
            score = item["SimilarityScore"]
            docs_and_scores.append((Document(page_content=text, metadata=item), score))
        return docs_and_scores

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        embeddings = self._embedding.embed_query(query)
        docs_and_scores = self._similarity_search_with_score(embeddings=embeddings, k=k)
        return docs_and_scores

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k=k)

        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        # Retrieves the docs with similarity scores
        docs = self._similarity_search_with_score(embeddings=embedding, k=fetch_k)

        # Re-ranks the docs using MMR
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )

        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        # compute the embeddings vector from the query string
        embeddings = self._embedding.embed_query(query)

        docs = self.max_marginal_relevance_search_by_vector(
            embeddings,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        return docs
