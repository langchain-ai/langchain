import logging
import uuid
from typing import Sequence

from bs4 import BeautifulSoup as Soup
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from propositional_retrieval.constants import DOCSTORE_ID_KEY
from propositional_retrieval.proposal_chain import proposition_chain
from propositional_retrieval.storage import get_multi_vector_retriever

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def add_documents(
    retriever,
    propositions: Sequence[Sequence[str]],
    docs: Sequence[Document],
    id_key: str = DOCSTORE_ID_KEY,
):
    doc_ids = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata["source"])) for doc in docs
    ]
    prop_docs = [
        Document(page_content=prop, metadata={id_key: doc_ids[i]})
        for i, props in enumerate(propositions)
        for prop in props
        if prop
    ]
    retriever.vectorstore.add_documents(prop_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))


def create_index(
    docs: Sequence[Document],
    indexer: Runnable,
    docstore_id_key: str = DOCSTORE_ID_KEY,
):
    """
    Create retriever that indexes docs and their propositions

    :param docs: Documents to index
    :param indexer: Runnable creates additional propositions per doc
    :param docstore_id_key: Key to use to store the docstore id
    :return: Retriever
    """
    logger.info("Creating multi-vector retriever")
    retriever = get_multi_vector_retriever(docstore_id_key)
    propositions = indexer.batch(
        [{"input": doc.page_content} for doc in docs], {"max_concurrency": 10}
    )

    add_documents(
        retriever,
        propositions,
        docs,
        id_key=docstore_id_key,
    )

    return retriever


if __name__ == "__main__":
    # For our example, we'll load docs from the web
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa
    from langchain_community.document_loaders.recursive_url_loader import (
        RecursiveUrlLoader,
    )  # noqa

    # The attention is all you need paper
    # Could add more parsing here, as it's very raw.
    loader = RecursiveUrlLoader(
        "https://ar5iv.labs.arxiv.org/html/1706.03762",
        max_depth=2,
        extractor=lambda x: Soup(x, "html.parser").text,
    )
    data = loader.load()
    logger.info(f"Loaded {len(data)} documents")

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    logger.info(f"Split into {len(all_splits)} documents")

    # Create retriever
    retriever_multi_vector_img = create_index(
        all_splits,
        proposition_chain,
        DOCSTORE_ID_KEY,
    )
