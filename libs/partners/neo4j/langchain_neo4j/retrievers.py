"""Neo4j retrievers."""

from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class Neo4jRetriever(BaseRetriever):
    # TODO: Replace all TODOs in docstring. See example docstring:
    # https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/tavily_search_api.py#L17
    """Neo4j retriever.

    # TODO: Replace with relevant packages, env vars, etc.
    Setup:
        Install ``langchain-neo4j`` and set environment variable ``NEO4J_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-neo4j
            export NEO4J_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args:
        arg 1: type
            description
        arg 2: type
            description

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain-neo4j import Neo4jRetriever

            retriever = Neo4jRetriever(
                # ...
            )

    Usage:
        .. code-block:: python

            query = "..."

            retriever.invoke(query)

        .. code-block:: none

            # TODO: Example output.

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

            chain.invoke("...")

        .. code-block:: none

             # TODO: Example output.

    """  # noqa: E501

    # TODO: This method must be implemented to retrieve documents.
    def _get_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError()
