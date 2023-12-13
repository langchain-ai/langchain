from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.max_compute import MaxComputeAPIWrapper


class MaxComputeLoader(BaseLoader):
    """Load from `Alibaba Cloud MaxCompute` table."""

    def __init__(
        self,
        query: str,
        api_wrapper: MaxComputeAPIWrapper,
        *,
        page_content_columns: Optional[Sequence[str]] = None,
        metadata_columns: Optional[Sequence[str]] = None,
    ):
        """Initialize Alibaba Cloud MaxCompute document loader.

        Args:
            query: SQL query to execute.
            api_wrapper: MaxCompute API wrapper.
            page_content_columns: The columns to write into the `page_content` of the
                Document. If unspecified, all columns will be written to `page_content`.
            metadata_columns: The columns to write into the `metadata` of the Document.
                If unspecified, all columns not added to `page_content` will be written.
        """
        self.query = query
        self.api_wrapper = api_wrapper
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns

    @classmethod
    def from_params(
        cls,
        query: str,
        endpoint: str,
        project: str,
        *,
        access_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        **kwargs: Any,
    ) -> MaxComputeLoader:
        """Convenience constructor that builds the MaxCompute API wrapper from
            given parameters.

        Args:
            query: SQL query to execute.
            endpoint: MaxCompute endpoint.
            project: A project is a basic organizational unit of MaxCompute, which is
                similar to a database.
            access_id: MaxCompute access ID. Should be passed in directly or set as the
                environment variable `MAX_COMPUTE_ACCESS_ID`.
            secret_access_key: MaxCompute secret access key. Should be passed in
                directly or set as the environment variable
                `MAX_COMPUTE_SECRET_ACCESS_KEY`.
        """
        api_wrapper = MaxComputeAPIWrapper.from_params(
            endpoint, project, access_id=access_id, secret_access_key=secret_access_key
        )
        return cls(query, api_wrapper, **kwargs)

    def lazy_load(self) -> Iterator[Document]:
        for row in self.api_wrapper.query(self.query):
            if self.page_content_columns:
                page_content_data = {
                    k: v for k, v in row.items() if k in self.page_content_columns
                }
            else:
                page_content_data = row
            page_content = "\n".join(f"{k}: {v}" for k, v in page_content_data.items())
            if self.metadata_columns:
                metadata = {k: v for k, v in row.items() if k in self.metadata_columns}
            else:
                metadata = {k: v for k, v in row.items() if k not in page_content_data}
            yield Document(page_content=page_content, metadata=metadata)

    def load(self) -> List[Document]:
        return list(self.lazy_load())
