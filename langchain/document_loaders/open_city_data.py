from typing import Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class OpenCityDataLoader(BaseLoader):
    """Loader that loads Open city data."""

    def __init__(self, city_id: str, dataset_id: str, limit: int):
        """Initialize with dataset_id"""
        """ Example: https://dev.socrata.com/foundry/data.sfgov.org/vw6y-z8j6 """
        """ e.g., city_id = data.sfgov.org """
        """ e.g., dataset_id = vw6y-z8j6 """
        self.city_id = city_id
        self.dataset_id = dataset_id
        self.limit = limit

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records."""

        import pandas as pd
        from sodapy import Socrata

        from langchain.document_loaders import DataFrameLoader

        client = Socrata(self.city_id, None)
        results = client.get(self.dataset_id, limit=self.limit)
        # Use dataframe intermediate to reduce code duplication
        results_df = pd.DataFrame.from_records(results)
        # Define a common column, as page_content is required
        # for Document and schemas differ per dataset. All other
        # fields will be added to Document metadata
        results_df["record"] = results_df.index + 1
        loader = DataFrameLoader(results_df, page_content_column="record")
        return loader.lazy_load()

    def load(self) -> List[Document]:
        """Load records."""

        return list(self.lazy_load())
