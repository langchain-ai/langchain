from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities.arxiv import ArxivAPIWrapper


class ArxivRetriever(BaseRetriever, ArxivAPIWrapper):
    """`Arxiv` retriever.

    Setup:
        Install ``arxiv``:

        .. code-block:: bash

            pip install -U arxiv

    Key init args:
        load_max_docs: int
            maximum number of documents to load
        get_ful_documents: bool
            whether to return full document text or snippets

    Instantiate:
        .. code-block:: python

            from langchain_community.retrievers import ArxivRetriever

            retriever = ArxivRetriever(
                load_max_docs=2,
                get_ful_documents=True,
            )

    Usage:
        .. code-block:: python

            docs = retriever.invoke("What is the ImageBind model?")
            docs[0].metadata

        .. code-block:: none

            {'Entry ID': 'http://arxiv.org/abs/2305.05665v2',
            'Published': datetime.date(2023, 5, 31),
            'Title': 'ImageBind: One Embedding Space To Bind Them All',
            'Authors': 'Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra'}

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

            chain.invoke("What is the ImageBind model?")

        .. code-block:: none

             'The ImageBind model is an approach to learn a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data...'
    """  # noqa: E501

    get_full_documents: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.get_full_documents:
            return self.load(query=query)
        else:
            return self.get_summaries_as_docs(query)
