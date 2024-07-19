"""Wrapper around Dria Retriever."""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities import DriaAPIWrapper


class DriaRetriever(BaseRetriever):
    """`Dria` retriever using the DriaAPIWrapper."""

    api_wrapper: DriaAPIWrapper

    def __init__(self, api_key: str, contract_id: Optional[str] = None, **kwargs: Any):
        """
        Initialize the DriaRetriever with a DriaAPIWrapper instance.

        Args:
            api_key: The API key for Dria.
            contract_id: The contract ID of the knowledge base to interact with.
        """
        api_wrapper = DriaAPIWrapper(api_key=api_key, contract_id=contract_id)
        super().__init__(api_wrapper=api_wrapper, **kwargs)  # type: ignore[call-arg]

    def create_knowledge_base(
        self,
        name: str,
        description: str,
        category: str = "Unspecified",
        embedding: str = "jina",
    ) -> str:
        """Create a new knowledge base in Dria.

        Args:
            name: The name of the knowledge base.
            description: The description of the knowledge base.
            category: The category of the knowledge base.
            embedding: The embedding model to use for the knowledge base.


        Returns:
            The ID of the created knowledge base.
        """
        response = self.api_wrapper.create_knowledge_base(
            name, description, category, embedding
        )
        return response

    def add_texts(
        self,
        texts: List,
    ) -> None:
        """Add texts to the Dria knowledge base.

        Args:
            texts: An iterable of texts and metadatas to add to the knowledge base.

        Returns:
            List of IDs representing the added texts.
        """
        data = [{"text": text["text"], "metadata": text["metadata"]} for text in texts]
        self.api_wrapper.insert_data(data)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents from Dria based on a query.

        Args:
            query: The query string to search for in the knowledge base.
            run_manager: Callback manager for the retriever run.

        Returns:
            A list of Documents containing the search results.
        """
        results = self.api_wrapper.search(query)
        docs = [
            Document(
                page_content=result["metadata"],
                metadata={"id": result["id"], "score": result["score"]},
            )
            for result in results
        ]
        return docs
