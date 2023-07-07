"""Wrapper around AwaDB for embedding vectors"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

# from pydantic import BaseModel, Field, root_validator


if TYPE_CHECKING:
    import awadb

logger = logging.getLogger()
DEFAULT_TOPN = 4


class AwaDB(VectorStore):
    """Interface implemented by AwaDB vector stores."""

    _DEFAULT_TABLE_NAME = "langchain_awadb"

    def __init__(
        self,
        table_name: str = _DEFAULT_TABLE_NAME,
        embedding: Optional[Embeddings] = None,
        log_and_data_dir: Optional[str] = None,
        client: Optional[awadb.Client] = None,
    ) -> None:
        """Initialize with AwaDB client."""

        try:
            import awadb
        except ImportError:
            raise ValueError(
                "Could not import awadb python package. "
                "Please install it with `pip install awadb`."
            )

        if client is not None:
            self.awadb_client = client
        else:
            if log_and_data_dir is not None:
                self.awadb_client = awadb.Client(log_and_data_dir)
            else:
                self.awadb_client = awadb.Client()

        if table_name == self._DEFAULT_TABLE_NAME:
            table_name += "_"
            table_name += str(uuid.uuid4()).split("-")[-1]

        self.awadb_client.Create(table_name)
        self.table2embeddings: dict[str, Embeddings] = {}
        if embedding is not None:
            self.table2embeddings[table_name] = embedding
        self.using_table_name = table_name

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        is_duplicate_texts: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            is_duplicate_texts: Optional whether to duplicate texts.
            kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if self.awadb_client is None:
            raise ValueError("AwaDB client is None!!!")

        embeddings = None
        if self.using_table_name in self.table2embeddings:
            embeddings = self.table2embeddings[self.using_table_name].embed_documents(
                list(texts)
            )

        return self.awadb_client.AddTexts(
            "embedding_text",
            "text_embedding",
            texts,
            embeddings,
            metadatas,
            is_duplicate_texts,
        )

    def load_local(
        self,
        table_name: str,
        **kwargs: Any,
    ) -> bool:
        if self.awadb_client is None:
            raise ValueError("AwaDB client is None!!!")

        return self.awadb_client.Load(table_name)

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_TOPN,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        if self.awadb_client is None:
            raise ValueError("AwaDB client is None!!!")

        embedding = None
        if self.using_table_name in self.table2embeddings:
            embedding = self.table2embeddings[self.using_table_name].embed_query(query)
        else:
            from awadb import llm_embedding

            llm = llm_embedding.LLMEmbedding()
            embedding = llm.Embedding(query)

        return self.similarity_search_by_vector(embedding, k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_TOPN,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores, normalized on a scale from 0 to 1.

        0 is dissimilar, 1 is most similar.
        """

        if self.awadb_client is None:
            raise ValueError("AwaDB client is None!!!")

        embedding = None
        if self.using_table_name in self.table2embeddings:
            embedding = self.table2embeddings[self.using_table_name].embed_query(query)
        else:
            from awadb import llm_embedding

            llm = llm_embedding.LLMEmbedding()
            embedding = llm.Embedding(query)

        results: List[Tuple[Document, float]] = []

        scores: List[float] = []
        retrieval_docs = self.similarity_search_by_vector(embedding, k, scores)

        L2_Norm = 0.0
        for score in scores:
            L2_Norm = L2_Norm + score * score

        L2_Norm = pow(L2_Norm, 0.5)
        doc_no = 0
        for doc in retrieval_docs:
            doc_tuple = (doc, 1 - (scores[doc_no] / L2_Norm))
            results.append(doc_tuple)
            doc_no = doc_no + 1

        return results

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = DEFAULT_TOPN,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores, normalized on a scale from 0 to 1.

        0 is dissimilar, 1 is most similar.
        """

        if self.awadb_client is None:
            raise ValueError("AwaDB client is None!!!")

        embedding = None
        if self.using_table_name in self.table2embeddings:
            embedding = self.table2embeddings[self.using_table_name].embed_query(query)

        show_results = self.awadb_client.Search(embedding, k)

        results: List[Tuple[Document, float]] = []

        if show_results.__len__() == 0:
            return results

        scores: List[float] = []
        retrieval_docs = self.similarity_search_by_vector(embedding, k, scores)

        L2_Norm = 0.0
        for score in scores:
            L2_Norm = L2_Norm + score * score

        L2_Norm = pow(L2_Norm, 0.5)
        doc_no = 0
        for doc in retrieval_docs:
            doc_tuple = (doc, 1 - scores[doc_no] / L2_Norm)
            results.append(doc_tuple)
            doc_no = doc_no + 1

        return results

    def similarity_search_by_vector(
        self,
        embedding: Optional[List[float]] = None,
        k: int = DEFAULT_TOPN,
        scores: Optional[list] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """

        if self.awadb_client is None:
            raise ValueError("AwaDB client is None!!!")

        results: List[Document] = []

        if embedding is None:
            return results

        show_results = self.awadb_client.Search(embedding, k)

        if show_results.__len__() == 0:
            return results

        for item_detail in show_results[0]["ResultItems"]:
            content = ""
            meta_data = {}
            for item_key in item_detail:
                if (
                    item_key == "Field@0"
                    and self.using_table_name in self.table2embeddings
                ):  # text for the document
                    content = item_detail[item_key]
                elif item_key == "embedding_text":
                    content = item_detail[item_key]
                elif (
                    item_key == "Field@1" or item_key == "text_embedding"
                ):  # embedding field for the document
                    continue
                elif item_key == "score":  # L2 distance
                    if scores is not None:
                        score = item_detail[item_key]
                        scores.append(score)
                else:
                    meta_data[item_key] = item_detail[item_key]
            results.append(Document(page_content=content, metadata=meta_data))
        return results

    def create_table(
        self,
        table_name: str,
        **kwargs: Any,
    ) -> bool:
        """Create a new table."""

        if self.awadb_client is None:
            return False

        ret = self.awadb_client.Create(table_name)

        if ret:
            self.using_table_name = table_name
        return ret

    def use(
        self,
        table_name: str,
        **kwargs: Any,
    ) -> bool:
        """Use the specified table. Don't know the tables, please invoke list_tables."""

        if self.awadb_client is None:
            return False

        ret = self.awadb_client.Use(table_name)
        if ret:
            self.using_table_name = table_name

        return ret

    def list_tables(
        self,
        **kwargs: Any,
    ) -> List[str]:
        """List all the tables created by the client."""

        if self.awadb_client is None:
            return []

        return self.awadb_client.ListAllTables()

    def get_current_table(
        self,
        **kwargs: Any,
    ) -> str:
        """Get the current table."""

        return self.using_table_name

    @classmethod
    def from_texts(
        cls: Type[AwaDB],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        table_name: str = _DEFAULT_TABLE_NAME,
        log_and_data_dir: Optional[str] = None,
        client: Optional[awadb.Client] = None,
        **kwargs: Any,
    ) -> AwaDB:
        """Create an AwaDB vectorstore from a raw documents.

        Args:
            texts (List[str]): List of texts to add to the table.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            table_name (str): Name of the table to create.
            log_and_data_dir (Optional[str]): Directory of logging and persistence.
            client (Optional[awadb.Client]): AwaDB client

        Returns:
            AwaDB: AwaDB vectorstore.
        """
        awadb_client = cls(
            table_name=table_name,
            embedding=embedding,
            log_and_data_dir=log_and_data_dir,
            client=client,
        )
        awadb_client.add_texts(texts=texts, metadatas=metadatas)
        return awadb_client

    @classmethod
    def from_documents(
        cls: Type[AwaDB],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        table_name: str = _DEFAULT_TABLE_NAME,
        log_and_data_dir: Optional[str] = None,
        client: Optional[awadb.Client] = None,
        **kwargs: Any,
    ) -> AwaDB:
        """Create an AwaDB vectorstore from a list of documents.

        If a log_and_data_dir specified, the table will be persisted there.

        Args:
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            table_name (str): Name of the table to create.
            log_and_data_dir (Optional[str]): Directory to persist the table.
            client (Optional[awadb.Client]): AwaDB client

        Returns:
            AwaDB: AwaDB vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            table_name=table_name,
            log_and_data_dir=log_and_data_dir,
            client=client,
        )
