"""Loader that loads documents from Psychic.dev."""
import json
from pathlib import Path
from typing import List

import pandas as pd

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class PsychicLoader(BaseLoader):
    """Loader that loads documents from Psychic.dev."""

    def __init__(self, api_key: str, connector_id: str, connection_id: str):
        """Initialize with API key, connector id, and connection id."""
        self.api_key = api_key

        try:
            from psychicapi import Psychic, ConnectorId  # noqa: F401
        except ImportError:
            raise ImportError(
                "`psychicapi` package not found, please run "
                "`pip install psychicapi`"
            )
        self.psychic = Psychic(secret_key=self.api_key)
        self.connector_id = ConnectorId(connector_id)
        self.connection_id = connection_id

    def load(self) -> List[Document]:
        """Load documents."""

        try: 
            psychic_docs = self.psychic.get_documents(self.connector_id, self.connection_id)
            return [
                Document(
                    page_content=doc["content"], 
                    metadata={"title": doc["title"], "source": doc["uri"]}
                ) for doc in psychic_docs
            ]
        except Exception as e:
            print(e)
            return []