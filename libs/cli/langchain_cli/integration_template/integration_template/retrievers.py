"""__ModuleName__ retrievers."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class __ModuleName__Retriever(BaseRetriever):
    # TODO: Replace all TODOs in docstring. See example docstring:
    # https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/tavily_search_api.py#L17
    """__ModuleName__ retriever.

    # TODO: Replace with relevant packages, env vars, etc.
    Setup:
        Install `__package_name__` and set environment variable
        `__MODULE_NAME___API_KEY`.

        ```bash
        pip install -U __package_name__
        export __MODULE_NAME___API_KEY="your-api-key"
        ```

    # TODO: Populate with relevant params.
    Key init args:
        arg 1: type
            description
        arg 2: type
            description

    # TODO: Replace with relevant init params.
    Instantiate:
        ```python
        from __package_name__ import __ModuleName__Retriever

        retriever = __ModuleName__Retriever(
            # ...
        )
        ```

    Usage:
        ```python
        query = "..."

        retriever.invoke(query)
        ```

        ```txt
        # TODO: Example output.
        ```

    Use within a chain:
        ```python
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_template(
            \"\"\"Answer the question based only on the context provided.

        Context: {context}

        Question: {question}\"\"\"
        )

        model = ChatOpenAI(model="gpt-3.5-turbo-0125")

        def format_docs(docs):
            return "\\n\\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        chain.invoke("...")
        ```

        ```
        # TODO: Example output.
        ```

    """

    k: int = 3

    # TODO: This method must be implemented to retrieve documents.
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k = kwargs.get("k", self.k)
        return [
            Document(page_content=f"Result {i} for query: {query}") for i in range(k)
        ]

    # optional: add custom async implementations here
    # async def _aget_relevant_documents(
    #     self,
    #     query: str,
    #     *,
    #     run_manager: AsyncCallbackManagerForRetrieverRun,
    #     **kwargs: Any,
    # ) -> list[Document]: ...
