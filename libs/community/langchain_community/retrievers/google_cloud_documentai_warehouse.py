"""Retriever wrapper for Google Cloud Document AI Warehouse."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env, pre_init

from langchain_community.utilities.vertexai import get_client_info

if TYPE_CHECKING:
    from google.cloud.contentwarehouse_v1 import (
        DocumentServiceClient,
        RequestMetadata,
        SearchDocumentsRequest,
    )
    from google.cloud.contentwarehouse_v1.services.document_service.pagers import (
        SearchDocumentsPager,
    )


@deprecated(
    since="0.0.32",
    removal="1.0",
    alternative_import="langchain_google_community.DocumentAIWarehouseRetriever",
)
class GoogleDocumentAIWarehouseRetriever(BaseRetriever):
    """A retriever based on Document AI Warehouse.

    Documents should be created and documents should be uploaded
        in a separate flow, and this retriever uses only Document AI
        schema_id provided to search for relevant documents.

    More info: https://cloud.google.com/document-ai-warehouse.
    """

    location: str = "us"
    """Google Cloud location where Document AI Warehouse is placed."""
    project_number: str
    """Google Cloud project number, should contain digits only."""
    schema_id: Optional[str] = None
    """Document AI Warehouse schema to query against.
    If nothing is provided, all documents in the project will be searched."""
    qa_size_limit: int = 5
    """The limit on the number of documents returned."""
    client: "DocumentServiceClient" = None  #: :meta private:

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates the environment."""
        try:
            from google.cloud.contentwarehouse_v1 import DocumentServiceClient
        except ImportError as exc:
            raise ImportError(
                "google.cloud.contentwarehouse is not installed."
                "Please install it with pip install google-cloud-contentwarehouse"
            ) from exc

        values["project_number"] = get_from_dict_or_env(
            values, "project_number", "PROJECT_NUMBER"
        )
        values["client"] = DocumentServiceClient(
            client_info=get_client_info(module="document-ai-warehouse")
        )
        return values

    def _prepare_request_metadata(self, user_ldap: str) -> "RequestMetadata":
        from google.cloud.contentwarehouse_v1 import RequestMetadata, UserInfo

        user_info = UserInfo(id=f"user:{user_ldap}")
        return RequestMetadata(user_info=user_info)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        request = self._prepare_search_request(query, **kwargs)
        response = self.client.search_documents(request=request)
        return self._parse_search_response(response=response)

    def _prepare_search_request(
        self, query: str, **kwargs: Any
    ) -> "SearchDocumentsRequest":
        from google.cloud.contentwarehouse_v1 import (
            DocumentQuery,
            SearchDocumentsRequest,
        )

        try:
            user_ldap = kwargs["user_ldap"]
        except KeyError:
            raise ValueError("Argument user_ldap should be provided!")

        request_metadata = self._prepare_request_metadata(user_ldap=user_ldap)
        schemas = []
        if self.schema_id:
            schemas.append(
                self.client.document_schema_path(
                    project=self.project_number,
                    location=self.location,
                    document_schema=self.schema_id,
                )
            )
        return SearchDocumentsRequest(
            parent=self.client.common_location_path(self.project_number, self.location),
            request_metadata=request_metadata,
            document_query=DocumentQuery(
                query=query, is_nl_query=True, document_schema_names=schemas
            ),
            qa_size_limit=self.qa_size_limit,
        )

    def _parse_search_response(
        self, response: "SearchDocumentsPager"
    ) -> List[Document]:
        documents = []
        for doc in response.matching_documents:
            metadata = {
                "title": doc.document.title,
                "source": doc.document.raw_document_path,
            }
            documents.append(
                Document(page_content=doc.search_text_snippet, metadata=metadata)
            )
        return documents
