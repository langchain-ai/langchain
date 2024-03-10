"""Wrapper around Embedchain Retriever."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class EmbedchainRetriever(BaseRetriever):
    """`Embedchain` retriever."""

    client: Any
    """Embedchain Pipeline."""

    @classmethod
    def create(cls, yaml_path: Optional[str] = None) -> EmbedchainRetriever:
        """
        Create a EmbedchainRetriever from a YAML configuration file.

        Args:
            yaml_path: Path to the YAML configuration file. If not provided,
                       a default configuration is used.

        Returns:
            An instance of EmbedchainRetriever.

        """
        from embedchain import Pipeline

        # Create an Embedchain Pipeline instance
        if yaml_path:
            client = Pipeline.from_config(yaml_path=yaml_path)
        else:
            client = Pipeline()
        return cls(client=client)

    def add_texts(
        self,
        texts: Iterable[str],
    ) -> List[str]:
        """Run more texts through the embeddings and add to the retriever.

        Args:
            texts: Iterable of strings/URLs to add to the retriever.

        Returns:
            List of ids from adding the texts into the retriever.
        """
        ids = []
        for text in texts:
            _id = self.client.add(text)
            ids.append(_id)
        return ids

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        res = self.client.search(query)

        docs = []
        for r in res:
            docs.append(
                Document(
                    page_content=r["context"],
                    metadata={
                        "source": r["metadata"]["url"],
                        "document_id": r["metadata"]["doc_id"],
                    },
                )
            )
        return docs
