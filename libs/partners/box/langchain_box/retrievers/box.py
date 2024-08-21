from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.pydantic_v1 import root_validator

from langchain_box.utilities import BoxAPIWrapper, BoxAuth

class BoxRetriever(BaseRetriever):
    """
    Box retriever.

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
    # TODO: update with Box code
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

    """String containing the Box Developer Token generated in the developer console"""
    box_developer_token: Optional[str] = None
    """Configured langchain_box.utilities.BoxAuth object"""
    box_auth: Optional[BoxAuth] = None
    """List[str] containing Box file ids"""
    box_file_ids: Optional[List[str]] = None

    box: Optional[BoxAPIWrapper]

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @root_validator(allow_reuse=True)
    def validate_box_loader_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        box = None

        """Validate that we have either a box_developer_token or box_auth."""
        if not values.get("box_auth") and not values.get("box_developer_token"):
            raise ValueError(
                "you must provide box_developer_token or a box_auth "
                "generated with langchain_box.utilities.BoxAuth"
            )

        box = BoxAPIWrapper(  # type: ignore[call-arg]
            box_developer_token=values.get("box_developer_token"),
            box_auth=values.get("box_auth"),
            character_limit=values.get("character_limit"),
        )

        values["box"] = box

        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        if self.box_file_ids:   # If using Box AI
            return self.box.ask_box_ai(query=query, box_file_ids=self.box_file_ids)
        else:                   # If using Search
            return self.box.search_box(query=query)