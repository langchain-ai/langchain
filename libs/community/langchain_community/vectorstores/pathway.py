"""
Pathway vector store server and client.


The PathwayVectorServer builds a pipeline which indexes all files in a given folder,
embeds them and builds a vector index. The pipeline reacts to changes in source files,
automatically updating appropriate index entries.

The PathwayVectorClient implements the LangChain VectorStore interface and queries the
PathwayVectorServer to retrieve up-to-date documents.

"""

from typing import TYPE_CHECKING, Callable, List, Optional

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from pathway.xpacks.llm import vector_store

class PathwayVectorServer:
    """
    Build an autoupdating document indexing pipeline
    for approximate nearest neighbor search.

    Args:
        embedder - embedding model e.g. OpenAIEmbeddings
        parser - callable that parses file contents into a list of documents
        splitter - document splitter, e.g. CharacterTextSplitter
    """

    def __init__(
        self,
        *docs,
        embedder: Embeddings,
        parser: Optional[Callable[[bytes], List[Document]]] = None,
        splitter: Optional[BaseDocumentTransformer] = None,
        **kwargs,
    ) -> None:
        try:
            from pathway.xpacks.llm import vector_store
        except ImportError:
            raise ImportError(
                "Could not import pathway python package. "
                "Please install it with `pip install pathway`."
            )

        generic_parser = None
        if parser:
            generic_parser = lambda x: [  # noqa
                (doc.page_content, doc.metadata) for doc in parser(x)
            ]

        generic_splitter = None
        if splitter:
            generic_splitter = lambda x: [  # noqa
                (doc.page_content, doc.metadata)
                for doc in splitter.transform_documents([Document(page_content=x)])
            ]
        generic_embedded = lambda x: embedder.embed_documents([x])[0]  # noqa

        self.vector_store_server = vector_store.VectorStoreServer(
            *docs,
            embedder=generic_embedded,
            parser=generic_parser,
            splitter=generic_splitter,
            **kwargs,
        )

    def run_server(
        self,
        host,
        port,
        threaded=False,
        with_cache=True,
        cache_backend=None,
    ):
        """
        Run the server and start answering queries.

        Args:
            - host: host to bind the HTTP listener
            - port: port to bind the HTTP listener
            - docs: pathway tables typically coming out of connectors which contain
              source documents.
            - threaded: if True, run in a thread. Else block computation
            - with_cache: if True, embedding requests for the same contents are cached
            - cache_backend: the backend to use for caching if it is enabled. The
              default is the disk cache, hosted locally in the folder ``./Cache``. You
              can use ``Backend`` class of the [`persistence API`]
              (/developers/api-docs/persistence-api/#pathway.persistence.Backend)
              to override it.

        Returns:
            If threaded, return the Thread object. Else, does not return.
        """
        try:
            import pathway as pw
        except ImportError:
            raise ImportError(
                "Could not import pathway python package. "
                "Please install it with `pip install pathway`."
            )
        if with_cache and cache_backend is None:
            cache_backend = pw.persistence.Backend.filesystem("./Cache")
        return self.vector_store_server.run_server(
            host,
            port,
            threaded=threaded,
            with_cache=with_cache,
            cache_backend=cache_backend,
        )


class PathwayVectorClient(VectorStore):
    """
    VectorStore connecting to PathwayVectorServer realtime data pipeline.
    """

    def __init__(
        self,
        host,
        port,
    ) -> None:
        self.client = vector_store.VectorStoreClient(host, port)

    def add_texts(
        self,
        *args,
        **kwargs,
    ) -> List[str]:
        """Pathway is not suitable for this method."""
        raise NotImplementedError(
            "Pathway vector store does not support adding or removing texts"
            " from client."
        )

    @classmethod
    def from_texts(cls, *args, **kwargs):
        raise NotImplementedError(
            "Pathway vector store does not support initializing from_texts."
        )

    def similarity_search(
        self, query: str, k: int = 4, metadata_filter: str | None = None
    ) -> List[Document]:
        rets = self.client(query=query, k=k, metadata_filter=metadata_filter)

        return [
            Document(page_content=ret["text"], metadata=ret["metadata"]) for ret in rets
        ]

    def get_vectorstore_statistics(self):
        """Fetch basic statistics about the Vector Store."""
        return self.client.get_vectorstore_statistics()
