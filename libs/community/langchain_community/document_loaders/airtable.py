from typing import Any, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class AirtableLoader(BaseLoader):
    """Load the `Airtable` tables."""

    def __init__(
        self, api_token: str, table_id: str, base_id: str, **kwargs: Any
    ) -> None:
        """Initialize with API token and the IDs for table and base.

        Args:
            api_token: Airtable API token.
            table_id: Airtable table ID.
            base_id:
            **kwargs: Additional parameters to pass to Table.all(). Refer to the
                pyairtable documentation for available options:
                https://pyairtable.readthedocs.io/en/latest/api.html#pyairtable.Table.all
        """  # noqa: E501
        self.api_token = api_token
        self.table_id = table_id
        self.base_id = base_id
        self.kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load Documents from table."""

        from pyairtable import Table

        table = Table(self.api_token, self.base_id, self.table_id)
        records = table.all(**self.kwargs)
        for record in records:
            metadata = {
                "source": self.base_id + "_" + self.table_id,
                "base_id": self.base_id,
                "table_id": self.table_id,
            }
            if "view" in self.kwargs:
                metadata["view"] = self.kwargs["view"]
            # Need to convert record from dict to str
            yield Document(page_content=str(record), metadata=metadata)
