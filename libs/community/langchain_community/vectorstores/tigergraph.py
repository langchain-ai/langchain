from __future__ import annotations

import enum
import warnings
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class SearchType(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    VECTOR = "vector"
    HYBRID = "hybrid"
    SIBLING = "sibling"


DEFAULT_SEARCH_TYPE = SearchType.HYBRID


class TigerGraphVector(VectorStore):
    """`TigerGraph` vector index.
    Used in conjunction with TigerGraph CoPilot SupportAI functionality.
    For more information, see https://docs.tigergraph.com/tg-copilot/current/intro/.

    To use, you should have the ``pyTigerGraph`` python package installed.

    Args:
        conn (TigerGraphConnection):
            A connection to a TigerGraph instance.
        search_type (SearchType):
            The search type to use.
            Defaults to "hybrid". Options are: "vector", "sibling", and "hybrid".
        search_index (Union[str, List[str], None]):
            The search index to use. Defaults to None.
    Example:
        .. code-block:: python
        from pyTigerGraph import TigerGraphConnection

        # We first create a connection to the database
        conn = TigerGraphConnection(
            host="https://<your_tigergraph_host>",
            username="<your_username>",
            password="<your_password>")

        # We then create a TigerGraphVector object
        # By default, the search type is "hybrid",
        # and the search indices are ["Entity", "Relationship",
        #                              "Document", "DocumentChunk"]
        from langchain_community.vectorstores import TigerGraphVector

        tg_vector = TigerGraphVector(
                        conn,
                        search_type="hybrid"
                    )

        # We can then add texts to the TigerGraph instance
        tg_vector.add_texts(["This is a test document"])

        # We can then search for similar documents
        results = tg_vector.search("This is a test document")
    """

    def __init__(
        self,
        conn: Any,
        search_type: SearchType = SearchType.HYBRID,
        search_index: Union[str, List[str], None] = None,
    ) -> None:
        try:
            from pyTigerGraph import TigerGraphConnection
        except ImportError:
            raise ImportError(
                "Could not import pyTigerGraph python package. "
                "Please install it with `pip install pyTigerGraph`."
            )

        search_index = (
            search_index
            if search_index is not None
            else [
                "Entity",
                "Relationship",
                "Document",
                "DocumentChunk",
            ]
        )

        if not isinstance(conn, TigerGraphConnection):
            msg = "**conn** parameter must inherit from TigerGraphConnection"
            raise TypeError(msg)

        self.search_index = search_index
        self.search_type = search_type
        self.conn = conn

        """Check if SupportAI is initialized"""
        v_types = self.conn.getVertexTypes()
        if (
            "Document" not in v_types
            and "DocumentChunk" not in v_types
            and "Entity" not in v_types
            and "Relationship" not in v_types
        ):
            warnings.warn(
                """SupportAI is not initialized for graph {}.
                            Initializing SupportAI.""".format(self.conn.graphname)
            )
            self.conn.ai.initializeSupportAI()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        upsert_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the TigerGraph CoPilot SupportAI.

        Args:
            texts: Iterable of strings to add to TigerGraph.
        Returns:
            Counts of documents, content edges, and content upserted.
        """
        import time

        cur_epoch = int(time.time())
        if ids:
            doc_vertices = [(id, {"epoch_added": cur_epoch}) for id in ids]
            doc_content_edges = [(id, id) for id in ids]
            content_vertices = [
                (id, {"text": text, "epoch_added": cur_epoch})
                for id, text in zip(ids, texts)
            ]
        else:
            # hash text to get unique ids
            import hashlib

            doc_vertices = [
                (
                    str(hashlib.md5(text.encode()).hexdigest()),
                    {"epoch_added": cur_epoch},
                )
                for text in texts
            ]
            doc_content_edges = [(ids[0], ids[0]) for ids in doc_vertices]
            content_vertices = [
                (doc_vertices[i][0], {"text": text, "epoch_added": cur_epoch})
                for i, text in enumerate(texts)
            ]

        documents_upserted = 0
        has_content_edges_upserted = 0
        content_upserted = 0

        if len(doc_vertices) < upsert_size:
            documents_upserted += self.conn.upsertVertices("Document", doc_vertices)
            has_content_edges_upserted += self.conn.upsertEdges(
                "Document", "HAS_CONTENT", "Content", doc_content_edges
            )
            content_upserted += self.conn.upsertVertices("Content", content_vertices)
        else:
            for i in range(0, len(doc_vertices), upsert_size):
                documents_upserted += self.conn.upsertVertices(
                    "Document", doc_vertices[i : i + upsert_size]
                )
                has_content_edges_upserted += self.conn.upsertEdges(
                    "Document",
                    "HAS_CONTENT",
                    "Content",
                    doc_content_edges[i : i + upsert_size],
                )
                content_upserted += self.conn.upsertVertices(
                    "Content", content_vertices[i : i + upsert_size]
                )

        if ids:
            return ids
        else:
            return [doc[0] for doc in doc_vertices]

    def add_documents_from_blob(
        self,
        data_source: str,
        data_source_config: Dict[str, Any],
        loader_config: Dict[str, Any],
        data_path: str,
        file_format: str = "json",
    ) -> Dict[str, str]:
        """
        Add documents from a blob data source to the TigerGraph CoPilot SupportAI.

        Args:
            data_source: Name of the data source. (e.g. "s3", "gcs", "azure")
            data_source_config: Configuration for the data source.
                (e.g. {"aws_access_key": "access_key", "aws_secret_key": "secret_key"})
            loader_config: Configuration for the loader, containing two keys:
                "doc_id_field" and "content_field", with values specifying the columns or fields in the source data.
            data_path: Path to the data, e.g. a bucket path.
            file_format: Format of the file. Defaults to "json".
        """  # noqa: E501

        res = self.conn.ai.createDocumentIngest(
            data_source=data_source,
            data_source_config=data_source_config,
            loader_config=loader_config,
            file_format=file_format,
        )

        return self.conn.ai.runDocumentIngest(
            res["load_job_id"], res["data_source_id"], data_path
        )

    def search(
        self, query: str, k: int = 4, *, params: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Run similarity search with TigerGraph CoPilot.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            params (Dict[str, Any]): Additional parameters for the search.

        Returns:
            List of Documents most similar to the query.
        """
        return self.similarity_search(query, k, params=params)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Run similarity search with TigerGraph CoPilot.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """

        params = params or {}
        if self.search_type == SearchType.VECTOR:
            method_params = {"index": self.search_index, "top_k": k, "withHyDE": False}

            if params.get("withHyDE"):
                method_params["withHyDE"] = params["withHyDE"]

            res = self.conn.ai.searchDocuments(
                query, method="vdb", method_parameters=method_params
            )[0]["@@final_retrieval"]

            docs = [
                Document(metadata={"source": key}, page_content=val)
                for key, val in res.items()
            ]
        elif self.search_type == SearchType.HYBRID:
            method_params = {
                "indices": self.search_index,
                "top_k": k,
                "num_hops": 2,
                "num_seen_min": 2,
            }

            if params.get("num_hops"):
                method_params["num_hops"] = params["num_hops"]
            if params.get("num_seen_min"):
                method_params["num_seen_min"] = params["num_seen_min"]
            res = self.conn.ai.searchDocuments(
                query, method="hnswoverlap", method_parameters=method_params
            )[0]["@@final_retrieval"]
            docs = []
            for retrieved_types in list(res.keys()):
                for key, val in res[retrieved_types].items():
                    docs.append(
                        Document(
                            metadata={"source": key, "index_source": retrieved_types},
                            page_content=val,
                        )
                    )
        elif self.search_type == SearchType.SIBLING:
            method_params = {
                "index": self.search_index,
                "top_k": k,
                "lookahead": 3,
                "lookback": 3,
                "withHyDE": False,
            }

            if params.get("lookahead"):
                method_params["lookahead"] = params["lookahead"]
            if params.get("lookback"):
                method_params["lookback"] = params["lookback"]
            if params.get("withHyDE"):
                method_params["withHyDE"] = params["withHyDE"]

            res = self.conn.ai.searchDocuments(
                query, method="sibling", method_parameters=method_params
            )[0]["@@sibling_set"]

            docs = []
            for root_doc, sibling_docs in res.items():
                for sibling_doc_id, sibling_doc_data in sibling_docs.items():
                    docs.append(
                        Document(
                            metadata={
                                "source": sibling_doc_id,
                                "root": root_doc,
                                "sibling_distance": sibling_doc_data["distance"],
                            },
                            page_content=sibling_doc_data["content"],
                        )
                    )
        else:
            raise ValueError("Invalid search type")

        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        upsert_size: int = 1000,
        conn: Any = None,
        **kwargs: Any,
    ) -> TigerGraphVector:
        """Create a TigerGraphVector from a list of texts."""
        tg = cls(conn)
        tg.add_texts(texts, ids=ids, upsert_size=upsert_size)
        return tg
