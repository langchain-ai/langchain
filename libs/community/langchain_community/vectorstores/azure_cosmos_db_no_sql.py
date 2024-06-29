from __future__ import annotations

import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from azure.cosmos.cosmos_client import CosmosClient

COMPARISONS_TO_SQL = {
    "$eq": "=",
    "$ne": "<>",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

IN_OPERATORS_TO_SQL = {
    "$in": "IN",
    "$nin": "NOT IN",
}

BETWEEN_OPERATOR = "$between"

LIKE_OPERATOR = "$like"

LOGICAL_OPERATORS_TO_SQL = {"$and": "AND", "$or": "OR"}

class AzureCosmosDBNoSqlVectorSearch(VectorStore):
    """`Azure Cosmos DB for NoSQL` vector store.

    To use, you should have both:
        - the ``azure-cosmos`` python package installed

    You can read more about vector search using AzureCosmosDBNoSQL here:
    https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
    """

    def __init__(
        self,
        *,
        cosmos_client: CosmosClient,
        embedding: Embeddings,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        create_container: bool = True,
    ):
        """
        Constructor for AzureCosmosDBNoSqlVectorSearch

        Args:
            cosmos_client: Client used to connect to azure cosmosdb no sql account.
            database_name: Name of the database to be created.
            container_name: Name of the container to be created.
            embedding: Text embedding model to use.
            vector_embedding_policy: Vector Embedding Policy for the container.
            indexing_policy: Indexing Policy for the container.
            cosmos_container_properties: Container Properties for the container.
            cosmos_database_properties: Database Properties for the container.
        """
        self._cosmos_client = cosmos_client
        self._database_name = database_name
        self._container_name = container_name
        self._embedding = embedding
        self._vector_embedding_policy = vector_embedding_policy
        self._indexing_policy = indexing_policy
        self._cosmos_container_properties = cosmos_container_properties
        self._cosmos_database_properties = cosmos_database_properties
        self._create_container = create_container

        if self._create_container:
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
            if self._cosmos_container_properties["partition_key"] is None:
                raise ValueError(
                    "partition_key cannot be null or empty for a container."
                )

        # Create the database if it already doesn't exist
        self._database = self._cosmos_client.create_database_if_not_exists(
            id=self._database_name,
            offer_throughput=self._cosmos_database_properties.get("offer_throughput"),
            session_token=self._cosmos_database_properties.get("session_token"),
            initial_headers=self._cosmos_database_properties.get("initial_headers"),
            etag=self._cosmos_database_properties.get("etag"),
            match_condition=self._cosmos_database_properties.get("match_condition"),
        )

        # Create the collection if it already doesn't exist
        self._container = self._database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=self._cosmos_container_properties["partition_key"],
            indexing_policy=self._indexing_policy,
            default_ttl=self._cosmos_container_properties.get("default_ttl"),
            offer_throughput=self._cosmos_container_properties.get("offer_throughput"),
            unique_key_policy=self._cosmos_container_properties.get(
                "unique_key_policy"
            ),
            conflict_resolution_policy=self._cosmos_container_properties.get(
                "conflict_resolution_policy"
            ),
            analytical_storage_ttl=self._cosmos_container_properties.get(
                "analytical_storage_ttl"
            ),
            computed_properties=self._cosmos_container_properties.get(
                "computed_properties"
            ),
            etag=self._cosmos_container_properties.get("etag"),
            match_condition=self._cosmos_container_properties.get("match_condition"),
            session_token=self._cosmos_container_properties.get("session_token"),
            initial_headers=self._cosmos_container_properties.get("initial_headers"),
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
    def _from_kwargs(
        cls,
        embedding: Embeddings,
        *,
        cosmos_client: CosmosClient,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        if kwargs:
            warnings.warn(
                "Method 'from_texts' of AzureCosmosDBNoSql vector "
                "store invoked with "
                f"unsupported arguments "
                f"({', '.join(sorted(kwargs))}), "
                "which will be ignored."
            )

        return cls(
            embedding=embedding,
            cosmos_client=cosmos_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            database_name=database_name,
            container_name=container_name,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        """Create an AzureCosmosDBNoSqlVectorSearch vectorstore from raw texts.

        Args:
            texts: the texts to insert.
            embedding: the embedding function to use in the store.
            metadatas: metadata dicts for the texts.
            **kwargs: you can pass any argument that you would
                to :meth:`~add_texts` and/or to the 'AstraDB' constructor
                (see these methods for details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an `AzureCosmosDBNoSqlVectorSearch` vectorstore.
        """
        vectorstore = AzureCosmosDBNoSqlVectorSearch._from_kwargs(embedding, **kwargs)
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        return vectorstore
    
    def _create_where_by_filter(self, filter):  # type: ignore[no-untyped-def]
        query_tuple = []
        where_str = ""
        if filter:
            where_str, query_tuple = self._process_filter_object(filter)
            where_str = " WHERE " + where_str
        return where_str, query_tuple
    
    def _process_filter_object(self, filter):  # type: ignore[no-untyped-def]
        query_tuple = []
        where_str = ""
        if filter:
            for i, key in enumerate(filter.keys()):
                filter_value = filter[key]
                if i != 0:
                    where_str += " AND "

                # Handling of 'special' boolean operators "$and", "$or"
                if key in LOGICAL_OPERATORS_TO_SQL:
                    logical_operator = LOGICAL_OPERATORS_TO_SQL[key]
                    logical_operands = filter_value
                    for j, logical_operand in enumerate(logical_operands):
                        if j != 0:
                            where_str += f" {logical_operator} "
                        (
                            where_str_logical,
                            query_tuple_logical,
                        ) = self._process_filter_object(logical_operand)
                        where_str += where_str_logical
                        query_tuple += query_tuple_logical
                    continue

                operator = "="
                sql_param = filter_value

                if isinstance(filter_value, bool):
                    sql_param = "true" if filter_value else "false"
                elif isinstance(filter_value, int) or isinstance(filter_value, str):
                    sql_param = filter_value
                elif isinstance(filter_value, Dict):
                    # Handling of 'special' operators starting with "$"
                    special_op = next(iter(filter_value))
                    special_val = filter_value[special_op]
                    # "$eq", "$ne", "$lt", "$lte", "$gt", "$gte"
                    if special_op in COMPARISONS_TO_SQL:
                        operator = COMPARISONS_TO_SQL[special_op]
                        if isinstance(special_val, bool):
                            sql_param = "true" if filter_value else "false"
                        elif isinstance(special_val, float):
                            sql_param = f"CAST({special_val} as float)"
                        elif isinstance(special_val, str):
                            sql_param = special_val
                    # "$between"
                    elif special_op == BETWEEN_OPERATOR:
                        between_from = special_val[0]
                        between_to = special_val[1]
                        operator = "BETWEEN"
                        sql_param = f"{between_from} AND {between_to}"
                    # "$like"
                    elif special_op == LIKE_OPERATOR:
                        operator = "LIKE"
                        sql_param = special_val
                    # "$in"
                    elif special_op in IN_OPERATORS_TO_SQL:
                        operator = IN_OPERATORS_TO_SQL[special_op]
                        if isinstance(special_val, list):
                            for i, list_entry in enumerate(special_val):
                                if i == 0:
                                    sql_param = "("
                                sql_param += list_entry
                                if i == (len(special_val) - 1):
                                    sql_param += ")"
                                else:
                                    sql_param += ","
                        else:
                            raise ValueError(
                                f"Unsupported value for {operator}: {special_val}"
                            )
                    else:
                        raise ValueError(f"Unsupported operator: {special_op}")
                else:
                    raise ValueError(
                        f"Unsupported filter data-type: {type(filter_value)}"
                    )
                where_str += f'c.{key} {operator} "{sql_param}"'

        return where_str, query_tuple

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
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        filter_query = ""
        if filter is not None:
            filter_query, _ = self._create_where_by_filter(filter)
            
        query = (
            f'''SELECT TOP {k} VALUE {{
                "document": c, 
                "SimilarityScore": VectorDistance(c.{self._embedding_key}, {embeddings}) 
            }} 
            FROM c 
            {filter_query} 
            ORDER BY VectorDistance(c.{self._embedding_key}, {embeddings})'''
        )

        docs_and_scores = []
        items = list(
            self._container.query_items(query=query, enable_cross_partition_query=True)
        )
        for item in items:
            document = item.get('document')
            text = document["text"]
            score = item.get('SimilarityScore')
            metadata = {k:v for k,v in document.items() if k not in ["id", "text", self._embedding_key] and not k.startswith("_")}
            docs_and_scores.append((Document(page_content=text, metadata=metadata), score))
        return docs_and_scores

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        embeddings = self._embedding.embed_query(query)
        docs_and_scores = self._similarity_search_with_score(embeddings=embeddings, k=k, filter=filter)
        return docs_and_scores

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)

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
