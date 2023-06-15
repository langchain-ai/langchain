from typing import Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class AirtableLoader(BaseLoader):
    """Loader that loads local airbyte json files."""

    def __init__(self, api_token: str, table_id: str, base_id: str):
        """Initialize with API token and the IDs for table and base"""
        self.api_token = api_token
        self.table_id = table_id
        self.base_id = base_id

    def lazy_load(self) -> Iterator[Document]:
        """Load Table."""

        from pyairtable import Table

        table = Table(self.api_token, self.base_id, self.table_id)
        records = table.all()
        for record in records:
            # Need to convert record from dict to str
            yield Document(
                page_content=str(record),
                metadata={
                    "source": self.base_id + "_" + self.table_id,
                    "base_id": self.base_id,
                    "table_id": self.table_id,
                },
            )

    def load(self) -> List[Document]:
        """Load Table."""
        return list(self.lazy_load())
