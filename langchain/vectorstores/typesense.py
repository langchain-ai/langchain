"""Wrapper around Typesense vector search"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

class Typesense(VectorStore):
    """Wrapper around Typesense vector search.

    To use, you should have the ``typesense`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import Typesense
            from langchain.embeddings.openai import OpenAIEmbeddings
            import typesense

            typesense_client = typesense.Client({
              "nodes": [{
                "host": "localhost",  # For Typesense Cloud use xxx.a1.typesense.net
                "port": "8108",       # For Typesense Cloud use 443
                "protocol": "http"    # For Typesense Cloud use https
              }],
              "api_key": "<API_KEY>",
              "connection_timeout_seconds": 2
            })
            typesense_collection_name = "products"

            embeddings = OpenAIEmbeddings()
            vectorstore = Typesense(typesense_client, typesense_collection_name, embeddings.embed_query, "text")
    """

    def __init__(
        self,
        typesense_client: Any,
        typesense_collection_name: str,
        embedding_function: Callable,
        text_key: str,
    ):
        """Initialize with Typesense client."""
        try:
            import typesense
        except ImportError:
            raise ValueError(
                "Could not import typesense python package. "
                "Please install it with `pip install typesense`."
            )
        if not isinstance(typesense_client, typesense.Client):
            raise ValueError(
                f"typesense_client should be an instance of typesense.Client, "
                f"got {type(typesense_client)}"
            )
        self.typesense_client = typesense_client
        self.typesense_collection_name = typesense_collection_name
        self._embedding_function = embedding_function
        self._text_key = text_key

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        from typesense.exceptions import (ObjectNotFound, ObjectAlreadyExists)
        import time

        # Embed and create the documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            embedding = self._embedding_function(text)
            metadata = metadatas[i] if metadatas else {}
            docs.append({
                "id": ids[i],
                "vec": embedding,
                "text": text,
                "metadata": metadata
            })

        # upsert to Typesense
        while True:
            try:
                self.typesense_client.collections[self.typesense_collection_name].documents.import_(
                    docs, {'action': 'upsert'}
                )
            except ObjectNotFound:
                # Create the collection if it doesn't already exist
                try:
                    self.typesense_client.collections.create({
                        "name": self.typesense_collection_name,
                        "fields": [
                            {
                                "name": "vec",
                                "type": "float[]",
                                "num_dim": len(docs[0]["vec"])
                            },
                            {
                                "name": "text",
                                "type": "string"
                            },
                            {
                                "name": ".*",
                                "type": "auto"
                            },
                        ]
                    })
                except ObjectAlreadyExists:
                    continue
                continue
            break
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = '',
    ) -> List[Tuple[Document, float]]:
        """Return typesense documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: typesense filter_by expression to filter documents on

        Returns:
            List of Documents most similar to the query and score for each
        """
        query_obj = self._embedding_function(query)
        docs = []
        results = self.typesense_client.collections[self.typesense_collection_name].documents.search({
            "q": "*",
            "vector_query": f'vec:([{",".join(query_obj)}], k:{k})',
            "filter_by": filter
        })
        for res in results["hits"]:
            metadata = res["metadata"]
            text = metadata.pop(self._text_key)
            docs.append((Document(page_content=text, metadata=metadata), res["vector_distance"]))
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = '',
    ) -> List[Document]:
        """Return typesense documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: typesense filter_by expression to filter documents on

        Returns:
            List of Documents most similar to the query and score for each
        """
        query_obj = self._embedding_function(query)
        docs = []
        results = self.typesense_client.collections[self.typesense_collection_name].documents.search({
            "q": "*",
            "vector_query": f'vec:([{",".join(query_obj)}], k:{k})',
            "filter_by": filter
        })
        for res in results["hits"]:
            metadata = res["metadata"]
            text = metadata.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        text_key: str = "text",
        typesense_collection_name: Optional[str] = None,
        typesense_client: Optional[Typesense.Client] = None,
        **kwargs: Any,
    ) -> Typesense.Client:
        """Construct Typesense wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Typesense collection

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Typesense
                from langchain.embeddings import OpenAIEmbeddings
                import typesense

                embeddings = OpenAIEmbeddings()

                typesense_client = typesense.Client({
                  "nodes": [{
                    "host": "localhost",  # For Typesense Cloud use xxx.a1.typesense.net
                    "port": "8108",       # For Typesense Cloud use 443
                    "protocol": "http"    # For Typesense Cloud use https
                  }],
                  "api_key": "<API_KEY>",
                  "connection_timeout_seconds": 2
                })

                Typesense.from_texts(
                    typesense_client,
                    texts,
                    embeddings,
                    index_name="langchain-demo"
                )
        """
        try:
            import typesense
        except ImportError:
            raise ValueError(
                "Could not import typesense python package. "
                "Please install it with `pip install typesense`."
            )

        from typesense.exceptions import (ObjectNotFound, ObjectAlreadyExists)

        _typesense_collection_name = typesense_collection_name or str(uuid.uuid4())

        for i in range(0, len(texts), batch_size):
            # set end position of batch
            i_end = min(i + batch_size, len(texts))
            # get batch of texts and ids
            lines_batch = texts[i:i_end]
            # create ids if not provided
            if ids:
                ids_batch = ids[i:i_end]
            else:
                ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
            # create embeddings
            embeds = embedding.embed_documents(lines_batch)
            # prep metadata and upsert batch
            if metadatas:
                metadata = metadatas[i:i_end]
            else:
                metadata = [{} for _ in range(i, i_end)]
            for j, line in enumerate(lines_batch):
                metadata[j][text_key] = line

            # Prep docs to upsert
            docs = []
            for i, text in enumerate(lines_batch):
                docs.append({
                    "id": ids_batch[i],
                    "vec": embeds[i],
                    "text": text,
                    "metadata": metadatas[i]
                })

            # upsert to Typesense
            while True:
                try:
                    typesense_client.collections[typesense_collection_name].documents.import_(
                        docs, {'action': 'upsert'}
                    )
                except ObjectNotFound:
                    # Create the collection if it doesn't already exist
                    try:
                        typesense_client.collections.create({
                            "name": typesense_collection_name,
                            "fields": [
                                {
                                    "name": "vec",
                                    "type": "float[]",
                                    "num_dim": len(docs[0]["vec"])
                                },
                                {
                                    "name": "text",
                                    "type": "string"
                                },
                                {
                                    "name": ".*",
                                    "type": "auto"
                                },
                            ]
                        })
                    except ObjectAlreadyExists:
                        continue
                    continue
                break
        return cls(typesense_client, embedding.embed_query, text_key)
