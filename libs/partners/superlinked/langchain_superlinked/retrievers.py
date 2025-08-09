"""Retriever for Superlinked."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, root_validator


class SuperlinkedRetriever(BaseRetriever):
    """
    Retriever for Superlinked, a library for building context-aware search and
    retrieval systems.

    Setup:
        Install the ``langchain-superlinked`` package and its peer ``superlinked``
        dependency.

        .. code-block:: bash

            pip install -U langchain-superlinked superlinked

    Key init args:
        sl_client: An instance of a Superlinked App (e.g., ``sl.InMemoryApp``).
        sl_query: A pre-constructed Superlinked QueryDescriptor object (from
                  ``sl.Query(...).find(...).similar(...)`` chain).
        page_content_field: The name of the field in a Superlinked result entry
                            that should be used as the ``page_content`` of a
                            LangChain Document.
        k: int
            Number of documents to return. Defaults to 4. Can be overridden at
            query time.
        query_text_param: Optional[str]
            The name of the parameter in the Superlinked Query to which the
            user's text query should be passed. Defaults to "query_text".
        metadata_fields: Optional[List[str]]
            A list of field names from a Superlinked result entry to be included
            in the metadata of a LangChain Document. If None, all fields except
            the ``page_content_field`` are included.

    Instantiate:
        .. code-block:: python

            import superlinked.framework as sl

            # 1. Define Superlinked Schema, Spaces, and Index
            class DocumentSchema(sl.Schema):
                id: sl.IdField
                text: sl.String

            doc_schema = DocumentSchema()
            text_space = sl.TextSimilaritySpace(text=doc_schema.text)
            doc_index = sl.Index([text_space])

            # 2. Define the Superlinked Query
            superlinked_query = (
                sl.Query(doc_index)
                .similar(text_space.text, sl.Param("query_text"))
                .select([doc_schema.text])
            )

            # 3. Set up the Superlinked App
            documents_data = [
                {"id": "1", "text": "The Eiffel Tower is in Paris."},
                {"id": "2", "text": "The Colosseum is in Rome."},
            ]

            # Create source and executor
            source = sl.InMemorySource(schema=doc_schema)
            executor = sl.InMemoryExecutor(sources=[source], indices=[doc_index])
            app = executor.run()

            # Add data to the source
            source.put(documents_data)

            # 4. Create the retriever
            retriever = SuperlinkedRetriever(
                sl_client=app,
                sl_query=superlinked_query,
                page_content_field="text",
                k=4  # Number of documents to return (optional, defaults to 4)
            )

    Usage:
        .. code-block:: python

            # Basic usage - returns up to k documents (default 4)
            retrieved_docs = retriever.invoke("Famous landmarks in France")

            # Override k at query time to return more/fewer documents
            retrieved_docs = retriever.invoke("Famous landmarks in France", k=2)

        .. code-block:: none

            [
                Document(
                    page_content='The Eiffel Tower is in Paris.',
                    metadata={'id': '1'}
                )
            ]

    Use within a chain:
        .. code-block:: python

            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI

            # Assuming 'retriever' is already instantiated
            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | ChatOpenAI()
                | StrOutputParser()
            )

            chain.invoke("What is a famous landmark in Paris?")

        .. code-block:: none

            'A famous landmark in Paris is the Eiffel Tower.'
    """

    sl_client: Any = Field(..., description="An instance of a Superlinked App.")
    sl_query: Any = Field(..., description="A Superlinked QueryDescriptor object.")
    query_text_param: str = Field(
        "query_text",
        description="The name of the parameter in the Superlinked Query to which "
        "the user's text query is passed.",
    )
    page_content_field: str = Field(
        ...,
        description="The name of the field in a Superlinked result entry that "
        "should be used as the page_content of a LangChain Document.",
    )
    metadata_fields: Optional[List[str]] = Field(
        None,
        description="A list of field names from a Superlinked result entry to be "
        "included in the metadata of a LangChain Document. If None, all fields "
        "except the page_content_field are included.",
    )
    k: int = Field(
        4,
        description="Number of documents to return. Can be overridden at query time.",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_superlinked_packages(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the superlinked package is installed and values are correct."""
        try:
            from superlinked.framework.dsl.app.app import App
            from superlinked.framework.dsl.query.query_descriptor import QueryDescriptor
        except ImportError:
            raise ImportError(
                "The 'superlinked' package is not installed. "
                "Please install it with 'pip install superlinked'"
            )

        if "sl_client" not in values or not isinstance(values.get("sl_client"), App):
            raise TypeError("sl_client must be a Superlinked App instance.")
        if "sl_query" not in values or not isinstance(
            values.get("sl_query"), QueryDescriptor
        ):
            raise TypeError("sl_query must be a Superlinked QueryDescriptor instance.")

        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """
        Retrieve relevant documents from Superlinked.

        Args:
            query: The user's text query.
            run_manager: The callback manager for the retriever run.
            **kwargs: Additional parameters to pass to the Superlinked query at runtime.

        Returns:
            A list of relevant LangChain Documents.
        """
        # Extract k parameter before building query_params
        k = kwargs.pop("k", self.k)

        query_params = kwargs.copy()
        query_params[self.query_text_param] = query

        try:
            results = self.sl_client.query(
                query_descriptor=self.sl_query, **query_params
            )
        except Exception:
            # Consider using proper logging in a production environment
            # Silently return empty list for now - in production, use proper logging
            return []

        documents: List[Document] = []
        for entry in results.entries:
            fields = entry.fields or {}

            if self.page_content_field not in fields:
                # If the designated content field is missing, skip this entry
                continue

            page_content = fields[self.page_content_field]
            metadata: Dict[str, Any] = {"id": entry.id}

            if self.metadata_fields is None:
                # Include all fields except the page_content_field
                for key, value in fields.items():
                    if key != self.page_content_field:
                        metadata[key] = value
            else:
                # Include only the specified fields
                for field_name in self.metadata_fields:
                    if field_name in fields:
                        metadata[field_name] = fields[field_name]

            documents.append(Document(page_content=page_content, metadata=metadata))

        # Apply k limit to the final results
        return documents[:k]
