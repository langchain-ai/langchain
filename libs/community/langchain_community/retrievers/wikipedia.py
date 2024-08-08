from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


class WikipediaRetriever(BaseRetriever, WikipediaAPIWrapper):
    """`Wikipedia API` retriever.

    Setup:
        Install the ``wikipedia`` dependency:

        .. code-block:: bash

            pip install -U wikipedia

    Instantiate:
        .. code-block:: python

            from langchain_community.retrievers import WikipediaRetriever

            retriever = WikipediaRetriever()

    Usage:
        .. code-block:: python

            docs = retriever.invoke("TOKYO GHOUL")
            print(docs[0].page_content[:100])

        .. code-block:: none

            Tokyo Ghoul (Japanese: 東京喰種（トーキョーグール）, Hepburn: Tōkyō Gūru) is a Japanese dark fantasy

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

            chain.invoke(
                "Who is the main character in `Tokyo Ghoul` and does he transform into a ghoul?"
            )

        .. code-block:: none

             'The main character in Tokyo Ghoul is Ken Kaneki, who transforms into a ghoul after receiving an organ transplant from a ghoul named Rize.'
    """  # noqa: E501

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.load(query=query)
