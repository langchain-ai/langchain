from __future__ import annotations

from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import RunnableBranch


def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    prompt: BasePromptTemplate,
) -> RetrieverOutputLike:
    """Create a chain that takes conversation history and returns documents.

    If there is no `chat_history`, then the `input` is just passed directly to the
    retriever. If there is `chat_history`, then the prompt and LLM will be used
    to generate a search query. That search query is then passed to the retriever.

    Args:
        llm: Language model to use for generating a search term given chat history
        retriever: RetrieverLike object that takes a string as input and outputs
            a list of Documents.
        prompt: The prompt used to generate the search query for the retriever.

    Returns:
        An LCEL Runnable. The runnable input must take in `input`, and if there
        is chat history should take it in the form of `chat_history`.
        The Runnable output is a list of Documents

    Example:
        .. code-block:: python

            # pip install -U langchain langchain-community

            from langchain_community.chat_models import ChatOpenAI
            from langchain.chains import create_history_aware_retriever
            from langchain import hub

            rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
            llm = ChatOpenAI()
            retriever = ...
            chat_retriever_chain = create_history_aware_retriever(
                llm, retriever, rephrase_prompt
            )

            chain.invoke({"input": "...", "chat_history": })

    """
    if "input" not in prompt.input_variables:
        raise ValueError(
            "Expected `input` to be a prompt variable, "
            f"but got {prompt.input_variables}"
        )

    retrieve_documents: RetrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["input"]) | retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")
    return retrieve_documents
