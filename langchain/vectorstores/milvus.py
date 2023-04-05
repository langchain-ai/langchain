"""Wrapper around the Milvus vector database."""
from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


class Milvus(VectorStore):
    """Wrapper around the Milvus vector database."""

    def __init__(
        self,
        embedding_function: Embeddings,
        connection_args: dict,
        collection_name: str,
        text_field: str,
    ):
        """Initialize wrapper around the milvus vector database.

        In order to use this you need to have `pymilvus` installed and a
        running Milvus instance.

        See the following documentation for how to run a Milvus instance:
        https://milvus.io/docs/install_standalone-docker.md

        Args:
            embedding_function (Embeddings): Function used to embed the text
            connection_args (dict): Arguments for pymilvus connections.connect()
            collection_name (str): The name of the collection to search.
            text_field (str): The field in Milvus schema where the
                original text is stored.
        """
        try:
            from pymilvus import Collection, DataType, connections
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        # Connecting to Milvus instance
        if not connections.has_connection("default"):
            connections.connect(**connection_args)
        self.embedding_func = embedding_function
        self.collection_name = collection_name

        self.text_field = text_field
        self.auto_id = False
        self.primary_field = None
        self.vector_field = None
        self.fields = []

        self.col = Collection(self.collection_name)
        schema = self.col.schema

        # Grabbing the fields for the existing collection.
        for x in schema.fields:
            self.fields.append(x.name)
            if x.auto_id:
                self.fields.remove(x.name)
            if x.is_primary:
                self.primary_field = x.name
            if x.dtype == DataType.FLOAT_VECTOR or x.dtype == DataType.BINARY_VECTOR:
                self.vector_field = x.name

        # Default search params when one is not provided.
        self.index_params = {
            "IVF_FLAT": {"params": {"nprobe": 10}},
            "IVF_SQ8": {"params": {"nprobe": 10}},
            "IVF_PQ": {"params": {"nprobe": 10}},
            "HNSW": {"params": {"ef": 10}},
            "RHNSW_FLAT": {"params": {"ef": 10}},
            "RHNSW_SQ": {"params": {"ef": 10}},
            "RHNSW_PQ": {"params": {"ef": 10}},
            "IVF_HNSW": {"params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"params": {"search_k": 10}},
        }

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        partition_name: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into Milvus.

        When using add_texts() it is assumed that a collecton has already
        been made and indexed. If metadata is included, it is assumed that
        it is ordered correctly to match the schema provided to the Collection
        and that the embedding vector is the first schema field.

        Args:
            texts (Iterable[str]): The text being embedded and inserted.
            metadatas (Optional[List[dict]], optional): The metadata that
                corresponds to each insert. Defaults to None.
            partition_name (str, optional): The partition of the collection
                to insert data into. Defaults to None.
            timeout: specified timeout.

        Returns:
            List[str]: The resulting keys for each inserted element.
        """
        insert_dict: Any = {self.text_field: list(texts)}
        try:
            insert_dict[self.vector_field] = self.embedding_func.embed_documents(
                list(texts)
            )
        except NotImplementedError:
            insert_dict[self.vector_field] = [
                self.embedding_func.embed_query(x) for x in texts
            ]
        # Collect the metadata into the insert dict.
        if len(self.fields) > 2 and metadatas is not None:
            for d in metadatas:
                for key, value in d.items():
                    if key in self.fields:
                        insert_dict.setdefault(key, []).append(value)
        # Convert dict to list of lists for insertion
        insert_list = [insert_dict[x] for x in self.fields]
        # Insert into the collection.
        res = self.col.insert(
            insert_list, partition_name=partition_name, timeout=timeout
        )
        # Flush to make sure newly inserted is immediately searchable.
        self.col.flush()
        return res.primary_keys

    def _worker_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Tuple[Document, Any, Any]]]:
        # Load the collection into memory for searching.
        self.col.load()
        # Decide to use default params if not passed in.
        if param is None:
            index_type = self.col.indexes[0].params["index_type"]
            param = self.index_params[index_type]
        # Embed the query text.
        data = [self.embedding_func.embed_query(query)]
        # Determine result metadata fields.
        output_fields = self.fields[:]
        output_fields.remove(self.vector_field)
        # Perform the search.
        res = self.col.search(
            data,
            self.vector_field,
            param,
            k,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            round_decimal=round_decimal,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            meta = {x: result.entity.get(x) for x in output_fields}
            ret.append(
                (
                    Document(page_content=meta.pop(self.text_field), metadata=meta),
                    result.distance,
                    result.id,
                )
            )

        return data[0], ret

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results.

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results ot return. Defaults to 4.
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            partition_names (List[str], optional): Partitions to search through.
                Defaults to None.
            round_decimal (int, optional): Round the resulting distance. Defaults
                to -1.
            timeout (int, optional): Amount to wait before timeout error. Defaults
                to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[float], List[Tuple[Document, any, any]]: search_embedding,
                (Document, distance, primary_field) results.
        """
        _, result = self._worker_search(
            query, k, param, expr, partition_names, round_decimal, timeout, **kwargs
        )
        return [(x, y) for x, y, _ in result]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            partition_names (List[str], optional): What partitions to search.
                Defaults to None.
            round_decimal (int, optional): Round the resulting distance. Defaults
                to -1.
            timeout (int, optional): Amount to wait before timeout error. Defaults
                to None.

        Returns:
            List[Document]: Document results for search.
        """
        data, res = self._worker_search(
            query,
            fetch_k,
            param,
            expr,
            partition_names,
            round_decimal,
            timeout,
            **kwargs,
        )
        # Extract result IDs.
        ids = [x for _, _, x in res]
        # Get the raw vectors from Milvus.
        vectors = self.col.query(
            expr=f"{self.primary_field} in {ids}",
            output_fields=[self.primary_field, self.vector_field],
        )
        # Reorganize the results from query to match result order.
        vectors = {x[self.primary_field]: x[self.vector_field] for x in vectors}
        search_embedding = data
        ordered_result_embeddings = [vectors[x] for x in ids]
        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(search_embedding), ordered_result_embeddings, k=k
        )
        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            if x == -1:
                break
            else:
                ret.append(res[x][0])
        return ret

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            partition_names (List[str], optional): What partitions to search.
                Defaults to None.
            round_decimal (int, optional): What decimal point to round to.
                Defaults to -1.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.

        Returns:
            List[Document]: Document results for search.
        """
        _, docs_and_scores = self._worker_search(
            query, k, param, expr, partition_names, round_decimal, timeout, **kwargs
        )
        return [doc for doc, _, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Milvus:
        """Create a Milvus collection, indexes it with HNSW, and insert data.

        Args:
            texts (List[str]): Text to insert.
            embedding (Embeddings): Embedding function to use.
            metadatas (Optional[List[dict]], optional): Dict metatadata.
                Defaults to None.

        Returns:
            VectorStore: The Milvus vector store.
        """
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
            )
            from pymilvus.orm.types import infer_dtype_bydata
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        # Connect to Milvus instance
        if not connections.has_connection("default"):
            connections.connect(**kwargs.get("connection_args", {"port": 19530}))
        # Determine embedding dim
        embeddings = embedding.embed_query(texts[0])
        dim = len(embeddings)
        # Generate unique names
        primary_field = "c" + str(uuid.uuid4().hex)
        vector_field = "c" + str(uuid.uuid4().hex)
        text_field = "c" + str(uuid.uuid4().hex)
        collection_name = "c" + str(uuid.uuid4().hex)
        fields = []
        # Determine metadata schema
        if metadatas:
            # Check if all metadata keys line up
            key = metadatas[0].keys()
            for x in metadatas:
                if key != x.keys():
                    raise ValueError(
                        "Mismatched metadata. "
                        "Make sure all metadata has the same keys and datatype."
                    )
            # Create FieldSchema for each entry in singular metadata.
            for key, value in metadatas[0].items():
                # Infer the corresponding datatype of the metadata
                dtype = infer_dtype_bydata(value)
                if dtype == DataType.UNKNOWN:
                    raise ValueError(f"Unrecognized datatype for {key}.")
                elif dtype == DataType.VARCHAR:
                    # Find out max length text based metadata
                    max_length = 0
                    for subvalues in metadatas:
                        max_length = max(max_length, len(subvalues[key]))
                    fields.append(
                        FieldSchema(key, DataType.VARCHAR, max_length=max_length + 1)
                    )
                else:
                    fields.append(FieldSchema(key, dtype))

        # Find out max length of texts
        max_length = 0
        for y in texts:
            max_length = max(max_length, len(y))
        # Create the text field
        fields.append(
            FieldSchema(text_field, DataType.VARCHAR, max_length=max_length + 1)
        )
        # Create the primary key field
        fields.append(
            FieldSchema(primary_field, DataType.INT64, is_primary=True, auto_id=True)
        )
        # Create the vector field
        fields.append(FieldSchema(vector_field, DataType.FLOAT_VECTOR, dim=dim))
        # Create the schema for the collection
        schema = CollectionSchema(fields)
        # Create the collection
        collection = Collection(collection_name, schema)
        # Index parameters for the collection
        index = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 8, "efConstruction": 64},
        }
        # Create the index
        collection.create_index(vector_field, index)
        # Create the VectorStore
        milvus = cls(
            embedding,
            kwargs.get("connection_args", {"port": 19530}),
            collection_name,
            text_field,
        )
        # Add the texts.
        milvus.add_texts(texts, metadatas)

        return milvus
