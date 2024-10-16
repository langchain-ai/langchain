from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import from_env
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_box.utilities import BoxAuth, BoxSearchOptions, _BoxAPIWrapper


class BoxRetriever(BaseRetriever):
    """Box retriever.

    `BoxRetriever` provides the ability to retrieve content from
    your Box instance in a couple of ways.

    1. You can use the Box full-text search to retrieve the
       complete document(s) that match your search query, as
       `List[Document]`
    2. You can use the Box AI Platform API to retrieve the results
       from a Box AI prompt. This can be a `Document` containing
       the result of the prompt, or you can retrieve the citations
       used to generate the prompt to include in your vectorstore.

    Setup:
        Install ``langchain-box``:

        .. code-block:: bash

            pip install -U langchain-box

    Instantiate:

        To use search:

        .. code-block:: python

            from langchain_box.retrievers import BoxRetriever

            retriever = BoxRetriever()

        To use Box AI:

        .. code-block:: python

            from langchain_box.retrievers import BoxRetriever

            file_ids=["12345","67890"]

            retriever = BoxRetriever(file_ids)


    Usage:
        .. code-block:: python

            retriever = BoxRetriever()
            retriever.invoke("victor")
            print(docs[0].page_content[:100])

        .. code-block:: none

            [
                Document(
                    metadata={
                        'source': 'url',
                        'title': 'FIVE_FEET_AND_RISING_by_Peter_Sollett_pdf'
                },
                    page_content='\\n3/20/23, 5:31 PM F...'
                )
            ]

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            retriever = BoxRetriever(box_developer_token=box_developer_token, character_limit=10000)

            context="You are an actor reading scripts to learn about your role in an upcoming movie."
            question="describe the character Victor"

            prompt = ChatPromptTemplate.from_template(
                \"""Answer the question based only on the context provided.

                Context: {context}

                Question: {question}\"""
            )

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("Victor")  # search query to find files in Box
            )

        .. code-block:: none

            'Victor is a skinny 12-year-old with sloppy hair who is seen
            sleeping on his fire escape in the sun. He is hesitant to go to
            the pool with his friend Carlos because he is afraid of getting
            in trouble for not letting his mother cut his hair. Ultimately,
            he decides to go to the pool with Carlos.'
    """  # noqa: E501

    box_developer_token: Optional[str] = Field(
        default_factory=from_env("BOX_DEVELOPER_TOKEN", default=None)
    )

    box_auth: Optional[BoxAuth] = None
    """Configured 
       `BoxAuth <https://python.langchain.com/v0.2/api_reference/box/utilities/langchain_box.utilities.box.BoxAuth.html>`_ 
       object"""

    box_file_ids: Optional[List[str]] = None
    """List[str] containing Box file ids"""

    character_limit: Optional[int] = -1
    """character_limit is an int that caps the number of characters to
       return per document."""

    box_search_options: Optional[BoxSearchOptions] = None
    """Search options to configure BoxRetriever to narrow search results."""

    answer: Optional[bool] = True
    """When using Box AI, return the answer to the prompt as a `Document` 
       object. Returned as `List[Document`]. Default is `True`."""

    citations: Optional[bool] = False
    """When using Box AI, return the citations from to the prompt as 
       `Document` objects. Can be used with answer. Returned as `List[Document`].
       Default is `False`."""

    _box: Optional[_BoxAPIWrapper]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def validate_box_loader_inputs(self) -> Self:
        _box = None

        """Validate that we have either a box_developer_token or box_auth."""
        if not self.box_auth and not self.box_developer_token:
            raise ValueError(
                "you must provide box_developer_token or a box_auth "
                "generated with langchain_box.utilities.BoxAuth"
            )

        _box = _BoxAPIWrapper(  # type: ignore[call-arg]
            box_developer_token=self.box_developer_token,
            box_auth=self.box_auth,
            character_limit=self.character_limit,
            box_search_options=self.box_search_options,
        )

        self._box = _box

        return self

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.box_file_ids:  # If using Box AI
            return self._box.ask_box_ai(  #  type: ignore[union-attr]
                query=query,
                box_file_ids=self.box_file_ids,
                answer=self.answer,  #  type: ignore[arg-type]
                citations=self.citations,  #  type: ignore[arg-type]
            )
        else:  # If using Search
            return self._box.search_box(query=query)  #  type: ignore[union-attr]
