"""Wrapper around Pinecone vector database."""
from __future__ import annotations
import itertools

import uuid
from typing import Any, Iterable, List, Optional, Tuple, cast

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

from tqdm.auto import tqdm


class Pinecone(VectorStore):
    """Wrapper around Pinecone vector database.

    To use, you should have the ``pinecone-client`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import Pinecone
            from langchain.embeddings.openai import OpenAIEmbeddings
            import pinecone

            pinecone.init(api_key="***", environment="us-west1-gcp")
            index = pinecone.Index("langchain-demo")
            embeddings = OpenAIEmbeddings()
            vectorstore = Pinecone(index, embeddings, "text")
    """

    def __init__(
        self,
        index: Any,
        embedding: Embeddings,
        text_key: str,
        namespace: Optional[str] = None,
    ):
        """Initialize with Pinecone client."""
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone-client`."
            )
        if not isinstance(index, pinecone.index.Index):
            raise ValueError(
                f"client should be an instance of pinecone.index.Index, "
                f"got {type(index)}"
            )
        self._index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            namespace: Optional pinecone namespace to add the texts to.
            batch_size: Optional batch size to use when embedding texts. Requires
                embedding function to support batching, during vector store
                initialization, can usually switch `embed.embed_query` to
                `embed.embed_documents` to enable batching.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        if namespace is None:
            namespace = self._namespace
        # Embed and create the documents
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        ids_iter = iter(ids)
        metadatas = metadatas or [{}]*len(ids)
        metadatas_iter = iter(metadatas)
        # Initialize the tqdm progress bar
        progress_bar = tqdm(total=len(ids), desc="Processing texts")
        # Process incoming texts in batches
        texts_iter = iter(texts)
        is_list = isinstance(texts, list)
        while True:
            if is_list:
                texts_list = cast(List[str], texts)
                text_batch = texts_list[:batch_size]
                texts = texts_list[batch_size:]
            else:
                text_batch = list(itertools.islice(texts_iter, batch_size))
            if not text_batch:
                break
            ids_batch = list(itertools.islice(ids_iter, len(text_batch)))
            metadata_batch = list(itertools.islice(metadatas_iter, len(text_batch)))
            metadata_batch = [{
                **metadata, self._text_key: text
            } for metadata, text in zip(metadata_batch, text_batch)]
            docs_batch: list = list(zip(
                ids_batch, self._embedding.embed_documents(text_batch), metadata_batch
            ))
            # upsert to Pinecone
            self._index.upsert(vectors=docs_batch, namespace=namespace)
            # Update the progress bar
            progress_bar.update(len(text_batch))
        # Close the progress bar
        progress_bar.close()
        return ids


    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return pinecone documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        if namespace is None:
            namespace = self._namespace
        query_obj = self._embedding.embed_query(query)
        docs = []
        results = self._index.query(
            [query_obj],
            top_k=k,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        for res in results["matches"]:
            metadata = res["metadata"]
            text = metadata.pop(self._text_key)
            docs.append((Document(page_content=text, metadata=metadata), res["score"]))
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return pinecone documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        if namespace is None:
            namespace = self._namespace
        query_obj = self._embedding.embed_query(query)
        docs = []
        results = self._index.query(
            [query_obj],
            top_k=k,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        for res in results["matches"]:
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
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> Pinecone:
        """Construct Pinecone wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Pinecone index

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Pinecone
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                pinecone = Pinecone.from_texts(
                    texts,
                    embeddings,
                    index_name="langchain-demo"
                )
        """
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone-client`."
            )
        _index_name = index_name or str(uuid.uuid4())
        indexes = pinecone.list_indexes()  # checks if provided index exists
        if _index_name in indexes:
            index = pinecone.Index(_index_name)
        else:
            index = None
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
            to_upsert = zip(ids_batch, embeds, metadata)
            # Create index if it does not exist
            if index is None:
                pinecone.create_index(_index_name, dimension=len(embeds[0]))
                index = pinecone.Index(_index_name)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert), namespace=namespace)
        return cls(index, embedding, text_key, namespace)

    @classmethod
    def from_existing_index(
        cls,
        index_name: str,
        embedding: Embeddings,
        text_key: str = "text",
        namespace: Optional[str] = None,
    ) -> Pinecone:
        """Load pinecone vectorstore from index name."""
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone-client`."
            )

        return cls(
            pinecone.Index(index_name), embedding, text_key, namespace
        )
