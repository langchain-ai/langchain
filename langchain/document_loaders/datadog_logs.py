from curses import meta
from typing import TYPE_CHECKING, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    from datadog_api_client import ApiClient, Configuration


class DatadogLogLoader(BaseLoader):
    """Loads a query result from Datadog into a list of documents.

    Logs are written into the `page_content` and into the `metadata`.
    """

    def __init__(
        self,
        query: str,
        api_key: str,
        app_key: str,
        from_time: Optional[int] = None,
        to_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        """
        Initialize Datadog document loader.

        Args:
            query: The query to run in Datadog.
            api_key: The Datadog API key.
            app_key: The Datadog APP key.
            from_time: Optional. The start of the time range to query. 
                Supports date math and regular timestamps (milliseconds) like '1688732708951'
            to_time: Optional. The end of the time range to query.
                Supports date math and regular timestamps (milliseconds) like '1688732708951'
            limit: Optional. The maximum number of logs to return.
        """
        self.query = query
        self.configuration = Configuration()
        self.configuration.api_key["apiKeyAuth"] = api_key
        self.configuration.api_key["appKeyAuth"] = app_key
        self.from_time = from_time
        self.to_time = to_time
        self.limit = limit if limit is not None else 100
    
    def parse_log(self, log: dict) -> Document:
        """
        Create Document objects from Datadog log items.
        """
        attributes = log.get("attributes", {})
        metadata = {
            "id": log.get("id", ""),
            "status": attributes.get("status"),
            "service": attributes.get("service", ""),
            "tags": attributes.get("tags", []),
            "timestamp": attributes.get("timestamp", ""),
        }

        message = attributes.get("message", "")
        inside_attributes = attributes.get("attributes", {})
        content_dict = {**inside_attributes, "message": message}
        content = ', '.join(f'{k}: {v}' for k, v in content_dict.items())
        return Document(page_content=content, metadata=metadata)
    
    def load(self) -> List[Document]:
        """
        Get logs from Datadog.

        Returns:
            A list of Document objects.
                - page_content
                - metadata
                    - id
                    - service
                    - status
                    - tags
                    - timestamp
        """
        try:
            from datadog_api_client.v2.api.logs_api import LogsApi
            from datadog_api_client.v2.model.logs_list_request import LogsListRequest
            from datadog_api_client.v2.model.logs_list_request_page import \
                LogsListRequestPage
            from datadog_api_client.v2.model.logs_query_filter import LogsQueryFilter
            from datadog_api_client.v2.model.logs_sort import LogsSort
        except ImportError as ex:
            raise ImportError(
                "Could not import datadog_api_client python package. "
                "Please install it with `pip install datadog_api_client`."
            ) from ex
        
        filter_params = {
            'query': self.query,
        }

        # Only add 'from' and 'to' parameters if they are not None
        if self.from_time is not None:
            filter_params['_from'] = f"{self.from_time}"
        if self.to_time is not None:
            filter_params['to'] = f"{self.to_time}"
    
        body = LogsListRequest(
            filter=LogsQueryFilter(**filter_params),
            sort=LogsSort.TIMESTAMP_ASCENDING,
            page=LogsListRequestPage(
                limit=self.limit,
            ),
        )

        with ApiClient(configuration=self.configuration) as api_client:
            api_instance = LogsApi(api_client)
            response = api_instance.list_logs(body=body).to_dict()
        
        docs: List[Document] = []
        for row in response['data']:
            docs.append(self.parse_log(row))

        return docs
