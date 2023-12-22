from typing import Iterator, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class OpenCityDataLoader(BaseLoader):
    """Load from `Open City`."""

    def __init__(self, city_id: str, dataset_id: str, limit: int):
        """Initialize with dataset_id.
        Example: https://dev.socrata.com/foundry/data.sfgov.org/vw6y-z8j6
        e.g., city_id = data.sfgov.org
        e.g., dataset_id = vw6y-z8j6

        Args:
            city_id: The Open City city identifier.
            dataset_id: The Open City dataset identifier.
            limit: The maximum number of documents to load.
        """
        self.city_id = city_id
        self.dataset_id = dataset_id
        self.limit = limit

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records."""

        from sodapy import Socrata

        client = Socrata(self.city_id, None)
        results = client.get(self.dataset_id, limit=self.limit)
        for record in results:
            yield Document(
                page_content=str(record),
                metadata={
                    "source": self.city_id + "_" + self.dataset_id,
                },
            )

    def load(self) -> List[Document]:
        """Load records."""

        return list(self.lazy_load())
