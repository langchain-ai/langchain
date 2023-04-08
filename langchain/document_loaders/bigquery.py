from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class BigQueryLoader(BaseLoader):
    """Loads a query result from BigQuery into a list of documents.

    Each document represents one row of the result. The `page_content_columns`
    are written into the `page_content` of the document. The `metadata_columns`
    are written into the `metadata` of the document. By default, all columns
    are written into the `page_content` and none into the `metadata`.
    """

    def __init__(
        self,
        query: str,
        project: Optional[str] = None,
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        self.query = query
        self.project = project
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns

    def load(self) -> List[Document]:
        try:
            from google.cloud import bigquery
        except ImportError as ex:
            raise ValueError(
                "Could not import google-cloud-bigquery python package. "
                "Please install it with `pip install google-cloud-bigquery`."
            ) from ex

        bq_client = bigquery.Client(self.project)
        query_result = bq_client.query(self.query).result()
        docs: List[Document] = []

        page_content_columns = self.page_content_columns
        metadata_columns = self.metadata_columns

        if page_content_columns is None:
            page_content_columns = [column.name for column in query_result.schema]
        if metadata_columns is None:
            metadata_columns = []

        for row in query_result:
            page_content = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in page_content_columns
            )
            metadata = {k: v for k, v in row.items() if k in metadata_columns}
            doc = Document(page_content=page_content, metadata=metadata)
            docs.append(doc)

        return docs
