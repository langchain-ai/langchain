import uuid
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from azure.cosmos.container import ContainerProxy
from azure.cosmos.cosmos_client import CosmosClient
from azure.cosmos.database import DatabaseProxy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance


# You can read more about vector search using AzureCosmosDBNoSQL here.
# https://aka.ms/CosmosVectorSearch
class AzureCosmosDBNoSqlVectorSearch(VectorStore):
    cosmos_client: CosmosClient = None
    database: DatabaseProxy
    container: ContainerProxy
    database_name: str = None
    container_name: str = None
    partition_key: str = None
    embedding: Embeddings = None
    vector_embedding_policy: [Dict[str, Any]] = None
    indexing_policy: [Dict[str, Any]] = None
    cosmos_container_properties: [Dict[str, Any]] = None

    def __init__(
            self,
            cosmos_client: CosmosClient,
            database_name: str,
            container_name: str,
            partition_key: str,
            embedding: Embeddings,
            vector_embedding_policy: [Dict[str, Any]],
            indexing_policy: [Dict[str, Any]],
            cosmos_container_properties: [Dict[str, Any]],
    ):
        if (indexing_policy["vectorIndexes"] is None
                or len(indexing_policy["vectorIndexes"]) == 0):
            raise ValueError("vectorIndexes cannot be null "
                             "or empty in the indexing_policy.")
        if (vector_embedding_policy is None
                or len(vector_embedding_policy["vectorEmbeddings"]) == 0):
            raise ValueError("vectorEmbeddings cannot be null "
                             "or empty in the vector_embedding_policy.")

        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.container_name = container_name
        self.embedding = embedding
        self.partition_key = partition_key
        self.vector_embedding_policy = vector_embedding_policy
        self.indexing_policy = indexing_policy
        self.cosmos_container_properties = cosmos_container_properties

        # Create the database if it already doesn't exist
        self.database = self.cosmos_client.create_database_if_not_exists(
            id=self.database_name)

        # Create the collection if it already doesn't exist
        self.container = self.database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=self.cosmos_container_properties["partition_key"],
            indexing_policy=self.indexing_policy,
            vector_embedding_policy=self.vector_embedding_policy,
        )

    def add_texts(self, texts: Iterable[str],
                  metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        _metadatas: Union[List, Generator] = metadatas or ({} for _ in texts)

        return self._insert_texts(list(texts), _metadatas)

    def _insert_texts(self, texts: List[str],
                      metadatas: List[Dict[str, Any]]) -> List[str]:
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
        embeddings = self.embedding.embed_documents(texts)
        text_key = "text"
        embedding_key = self.vector_embedding_policy["vectorEmbeddings"][0]["path"][1:]
        to_insert = [
            {"id": str(uuid.uuid4()), text_key: t, embedding_key: embedding, **m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in CosmosDB No Sql
        doc_ids: List[str] = []
        for item in to_insert:
            created_doc = self.container.create_item(item)
            doc_ids.append(created_doc["id"])
        return doc_ids

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        cosmos_client: CosmosClient = None,
        database_name: str = None,
        container_name: str = None,
        partition_key: str = None,
        vector_embedding_policy: [Dict[str, Any]] = None,
        indexing_policy: [Dict[str, Any]] = None,
        cosmos_container_properties: [Dict[str, Any]] = None,
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
            raise ValueError("Must provide 'cosmos_container_properties' "
                             "named parameter.")

        vectorstore = cls(
            cosmos_client,
            database_name,
            container_name,
            partition_key,
            embedding,
            vector_embedding_policy,
            indexing_policy,
            cosmos_container_properties
        )
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            self.container.delete_item(document_id)
        return True

    def delete_document_by_id(self, document_id: Optional[str] = None) -> None:
        """Removes a Specific Document by id

        Args:
            document_id: The document identifier
        """
        if document_id is None:
            raise ValueError("No document ids provided to delete.")
        self.container.delete_item(document_id, partition_key=document_id)

    def _similarity_search_with_score(
            self,
            embeddings: List[float],
            k: int = 4,
    ) -> List[Tuple[Document, float]]:
        embedding_key = self.vector_embedding_policy["vectorEmbeddings"][0]["path"][1:]
        query = (
            "SELECT TOP {} c.id, c.{}, c.text, VectorDistance(c.{}, {}) AS "
            "SimilarityScore FROM c ORDER BY VectorDistance(c.{}, {})"
            .format(k, embedding_key, embedding_key, embeddings,
                    embedding_key, embeddings)
        )
        docs_and_scores = []
        items = list(self.container.query_items(query=query,
                                                enable_cross_partition_query=True))
        for item in items:
            text = item["text"]
            score = item["SimilarityScore"]
            docs_and_scores.append(
                (Document(page_content=text, metadata=item), score)
            )
        return docs_and_scores

    def similarity_search_with_score(self, query: str, k: int = 4,
                                     ) -> List[Tuple[Document, float]]:
        embeddings = self.embedding.embed_query(query)
        docs_and_scores = self._similarity_search_with_score(
            embeddings=embeddings,
            k=k
        )
        return docs_and_scores

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any
                          ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k=k
        )

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
        docs = self._similarity_search_with_score(
            embeddings=embedding,
            k=fetch_k
        )

        # Re-ranks the docs using MMR
        embedding_key = self.vector_embedding_policy["vectorEmbeddings"][0]["path"][1:]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [doc.metadata[embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult
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
        embeddings = self.embedding.embed_query(query)

        docs = self.max_marginal_relevance_search_by_vector(
            embeddings,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        return docs
