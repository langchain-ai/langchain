"""Wrapper around Pinecone vector database."""
from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore


class Pinecone(VectorStore):
    """Wrapper around Pinecone vector database.

    To use, you should have the ``pinecone-client`` python package installed.

    Example:
        .. code-block:: python

            import pinecone
            from langchain.vectorstores import Pinecone
            vectorstore = Pinecone(pinecone_index, embedding_function, "text_key")
    """

    def __init__(
        self,
        index: Any,
        embedding_function: Callable,
        text_key: str,
    ):
        """Initialize with Pinecone client."""
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please it install it with `pip install pinecone-client`."
            )
        if not isinstance(index, pinecone.index.Index):
            raise ValueError(
                f"client should be an instance of pinecone.index.Index, got {type(index)}"
            )
        self._index = index
        self._embedding_function = embedding_function
        self._text_key = text_key

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        """Initialize with Pinecone client."""
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please it install it with `pip install pinecone-client`."
            )
        # Embed and create the documents
        docs = []
        ids = []
        for i, text in enumerate(texts):
            id = str(uuid.uuid4())
            embedding = self._embedding_function(text)
            metadata = metadatas[i] if metadatas else {}
            metadata[self._text_key] = text
            docs.append((id, embedding, metadata))
            ids.append(id)
        # upsert to Pinecone
        self._index.upsert(vectors=docs)
        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Look up similar documents in pinecone."""
        query_obj = self._embedding_function(query)
        docs = []
        results = self._index.query([query_obj], top_k=5, include_metadata=True)
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
        batch_size: int = 32,
        text_key: str = "text",
        **kwargs: Any,
    ) -> Pinecone:
        """Construct Pinecone wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in Pinecone
            3. Adds the documents to the newly created Pinecone index

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Pinecone
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                pinecone = Pinecone.from_texts(
                    texts,
                    embeddings
                    api_key="PINECONE_API_KEY"
                )
        """
        api_key = get_from_dict_or_env(kwargs, "api_key", "PINECONE_API_KEY")
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone-client`."
            )
        try:
            pinecone.init(api_key=api_key, environment="us-west1-gcp")
        except ValueError as e:
            raise ValueError("Pinecone initialization failed. " f"Got error: {e} ")

        # Create first embedding to get correct dims
        res = embedding.embed_query(texts[0])
        index_name = uuid.uuid4().hex
        pinecone.create_index(index_name, dimension=len(res))
        # connect to index
        index = pinecone.Index(index_name)
        for i in range(0, len(texts), batch_size):
            # set end position of batch
            i_end = min(i + batch_size, len(texts))
            # get batch of texts and ids
            lines_batch = texts[i : i + batch_size]
            ids_batch = [str(n) for n in range(i, i_end)]
            # create embeddings
            embeds = embedding.embed_documents(lines_batch)
            # prep metadata and upsert batch
            if metadatas:
                metadata = metadatas[i : i + batch_size]
            else:
                metadata = [{} for _ in range(i, i_end)]
            for j, line in enumerate(lines_batch):
                metadata[j][text_key] = line
            to_upsert = zip(ids_batch, embeds, metadata)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert))
        return cls(index, embedding.embed_query, text_key)
