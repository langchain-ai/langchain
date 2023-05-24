from typing import List, Optional
from langchain.docstore.document import Document

from langchain.document_loaders.base import BaseLoader


class FaunaDBLoader(BaseLoader):
    def __init__(
        self,
        query: str,
        page_content_field: str,
        secrect: str,
        metadata_fields: Optional[List[str]] = None,
        after: Optional[str] = None,
    ):
        self.query = query
        self.page_content_field = page_content_field
        self.secrect = secrect
        self.metadata_fields = metadata_fields

    def load(self) -> List[Document]:
        try:
            from fauna import fql, Page
            from fauna.client import Client
            from fauna.encoding import QuerySuccess
        except ImportError:
            raise ValueError(
                "Could not import fauna python package. "
                "Please install it with `pip install fauna`."
            )
        documents = []
        # Create Fauna Client
        client = Client(secret=self.secrect) 
        # Run FQL Query
        response: QuerySuccess = client.query(fql(self.query))
        page: Page = response.data
        for result in page:
            if result is not None:
                document_dict = dict(result.items())
                page_content = ''
                for key, value in document_dict.items():
                    if key == self.page_content_field:
                        page_content = value
                document: Document = Document(
                    page_content=page_content,
                    metadata={"id": result.id, "ts": result.ts},
                )
                documents.append(document)
        if page.after is not None:
            documents.append(
                Document(
                    page_content="Next Page Exists",
                    metadata={"after": page.after},
                )
            )
        return documents