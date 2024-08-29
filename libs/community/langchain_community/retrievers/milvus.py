"""Milvus Retriever"""

import warnings
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever

from langchain_community.vectorstores.milvus import Milvus

# TODO: Update to MilvusClient + Hybrid Search when available


class MilvusRetriever(BaseRetriever):
    """Milvus API retriever.

    See detailed instructions here: https://python.langchain.com/v0.2/docs/integrations/retrievers/milvus_hybrid_search/

    Setup:
        Install ``langchain-milvus`` and other dependencies:

        .. code-block:: bash

            pip install -U pymilvus[model] langchain-milvus

    Key init args:
        collection: Milvus Collection

    Instantiate:
        .. code-block:: python

            retriever = MilvusCollectionHybridSearchRetriever(collection=collection)

    Usage:
        .. code-block:: python

            query = "What are the story about ventures?"

            retriever.invoke(query)

        .. code-block:: none

            [Document(page_content="In 'The Lost Expedition' by Caspian Grey...", metadata={'doc_id': '449281835035545843'}),
            Document(page_content="In 'The Phantom Pilgrim' by Rowan Welles...", metadata={'doc_id': '449281835035545845'}),
            Document(page_content="In 'The Dreamwalker's Journey' by Lyra Snow..", metadata={'doc_id': '449281835035545846'})]

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("What novels has Lila written and what are their contents?")

        .. code-block:: none

             "Lila Rose has written 'The Memory Thief,' which follows a charismatic thief..."

    """  # noqa: E501

    embedding_function: Embeddings
    collection_name: str = "LangChainCollection"
    collection_properties: Optional[Dict[str, Any]] = None
    connection_args: Optional[Dict[str, Any]] = None
    consistency_level: str = "Session"
    search_params: Optional[dict] = None

    store: Milvus
    retriever: BaseRetriever

    @root_validator(pre=True)
    def create_retriever(cls, values: Dict) -> Dict:
        """Create the Milvus store and retriever."""
        values["store"] = Milvus(
            values["embedding_function"],
            values["collection_name"],
            values["collection_properties"],
            values["connection_args"],
            values["consistency_level"],
        )
        values["retriever"] = values["store"].as_retriever(
            search_kwargs={"param": values["search_params"]}
        )
        return values

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """Add text to the Milvus store

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.store.add_texts(texts, metadatas)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        return self.retriever.invoke(
            query, run_manager=run_manager.get_child(), **kwargs
        )


def MilvusRetreiver(*args: Any, **kwargs: Any) -> MilvusRetriever:
    """Deprecated MilvusRetreiver. Please use MilvusRetriever ('i' before 'e') instead.

    Args:
        *args:
        **kwargs:

    Returns:
        MilvusRetriever
    """
    warnings.warn(
        "MilvusRetreiver will be deprecated in the future. "
        "Please use MilvusRetriever ('i' before 'e') instead.",
        DeprecationWarning,
    )
    return MilvusRetriever(*args, **kwargs)
