from typing import Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class PsychicLoader(BaseLoader):
    """Load from `Psychic.dev`."""

    def __init__(
        self, api_key: str, account_id: str, connector_id: Optional[str] = None
    ):
        """Initialize with API key, connector id, and account id.

        Args:
            api_key: The Psychic API key.
            account_id: The Psychic account id.
            connector_id: The Psychic connector id.
        """

        try:
            from psychicapi import ConnectorId, Psychic  # noqa: F401
        except ImportError:
            raise ImportError(
                "`psychicapi` package not found, please run `pip install psychicapi`"
            )
        self.psychic = Psychic(secret_key=api_key)
        self.connector_id = ConnectorId(connector_id)
        self.account_id = account_id

    def lazy_load(self) -> Iterator[Document]:
        psychic_docs = self.psychic.get_documents(
            connector_id=self.connector_id, account_id=self.account_id
        )
        for doc in psychic_docs.documents:
            yield Document(
                page_content=doc["content"],
                metadata={"title": doc["title"], "source": doc["uri"]},
            )
