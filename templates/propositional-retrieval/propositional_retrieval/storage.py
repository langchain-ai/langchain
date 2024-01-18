import logging
from pathlib import Path

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def get_multi_vector_retriever(docstore_id_key: str):
    """Create the composed retriever object."""
    vectorstore = get_vectorstore()
    store = get_docstore()
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=docstore_id_key,
    )


def get_vectorstore(collection_name: str = "proposals"):
    """Get the vectorstore used for this example."""
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(Path(__file__).parent.parent / "chroma_db_proposals"),
        embedding_function=OpenAIEmbeddings(),
    )


def get_docstore():
    """Get the metadata store used for this example."""
    return LocalFileStore(
        str(Path(__file__).parent.parent / "multi_vector_retriever_metadata")
    )
