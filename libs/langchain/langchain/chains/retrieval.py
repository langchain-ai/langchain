from __future__ import annotations

from typing import Any, Dict, Union

from langchain_core.retrievers import (
    BaseRetriever,
    RetrieverOutput,
)
from langchain_core.runnables import Runnable, RunnablePassthrough


def create_retrieval_chain(
    retriever: Union[BaseRetriever, Runnable[dict, RetrieverOutput]],
    combine_docs_chain: Runnable[Dict[str, Any], str],
) -> Runnable:
    """Create retrieval chain that retrieves documents and then passes them on.

    Args:
        retriever: Retriever-like object that returns list of documents. Should
            either be a subclass of BaseRetriever or a Runnable that returns
            a list of documents. If a subclass of BaseRetriever, then it
            is expected that an `input` key be passed in - this is what
            is will be used to pass into the retriever. If this is NOT a
            subclass of BaseRetriever, then all the inputs will be passed
            into this runnable, meaning that runnable should take a dictionary
            as input.
        combine_docs_chain: Runnable that takes inputs and produces a string output.
            The inputs to this will be any original inputs to this chain, a new
            context key with the retrieved documents, and chat_history (if not present
            in the inputs) with a value of `[]` (to easily enable conversational
            retrieval.

    Returns:
        An LCEL Runnable. The Runnable return is a dictionary containing at the very
        least a `context` and `answer` key.

    Example:
        .. code-block:: python

            # pip install -U langchain langchain-community

            from langchain_community.chat_models import ChatOpenAI
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains import create_retrieval_chain
            from langchain import hub

            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            llm = ChatOpenAI()
            retriever = ...
            combine_docs_chain = create_stuff_documents_chain(
                llm, retrieval_qa_chat_prompt
            )
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            chain.invoke({"input": "..."})

    """
    if not isinstance(retriever, BaseRetriever):
        retrieval_docs: Runnable[dict, RetrieverOutput] = retriever
    else:
        retrieval_docs = (lambda x: x["input"]) | retriever

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
        ).assign(answer=combine_docs_chain)
    ).with_config(run_name="retrieval_chain")

    return retrieval_chain
