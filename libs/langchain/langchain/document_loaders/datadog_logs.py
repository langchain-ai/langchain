from datetime import datetime, timedelta
from typing import List, Optional

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader


class DatadogLogsLoader(BaseLoader):
    """Load `Datadog` logs.

    Logs are written into the `page_content` and into the `metadata`.
    """

    def __init__(
        self,
        query: str,
        api_key: str,
        app_key: str,
        from_time: Optional[int] = None,
        to_time: Optional[int] = None,
        limit: int = 100,
    ) -> None:
        """Initialize Datadog document loader.

        Requirements:
            - Must have datadog_api_client installed. Install with `pip install datadog_api_client`.

        Args:
            query: The query to run in Datadog.
            api_key: The Datadog API key.
            app_key: The Datadog APP key.
            from_time: Optional. The start of the time range to query.
                Supports date math and regular timestamps (milliseconds) like '1688732708951'
                Defaults to 20 minutes ago.
            to_time: Optional. The end of the time range to query.
                Supports date math and regular timestamps (milliseconds) like '1688732708951'
                Defaults to now.
            limit: The maximum number of logs to return.
                Defaults to 100.
        """  # noqa: E501
        try:
            from datadog_api_client import Configuration
        except ImportError as ex:
            raise ImportError(
                "Could not import datadog_api_client python package. "
                "Please install it with `pip install datadog_api_client`."
            ) from ex

        self.query = query
        configuration = Configuration()
        configuration.api_key["apiKeyAuth"] = api_key
        configuration.api_key["appKeyAuth"] = app_key
        self.configuration = configuration
        self.from_time = from_time
        self.to_time = to_time
        self.limit = limit

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
        content = ", ".join(f"{k}: {v}" for k, v in content_dict.items())
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
            from datadog_api_client import ApiClient
            from datadog_api_client.v2.api.logs_api import LogsApi
            from datadog_api_client.v2.model.logs_list_request import LogsListRequest
            from datadog_api_client.v2.model.logs_list_request_page import (
                LogsListRequestPage,
            )
            from datadog_api_client.v2.model.logs_query_filter import LogsQueryFilter
            from datadog_api_client.v2.model.logs_sort import LogsSort
        except ImportError as ex:
            raise ImportError(
                "Could not import datadog_api_client python package. "
                "Please install it with `pip install datadog_api_client`."
            ) from ex

        now = datetime.now()
        twenty_minutes_before = now - timedelta(minutes=20)
        now_timestamp = int(now.timestamp() * 1000)
        twenty_minutes_before_timestamp = int(twenty_minutes_before.timestamp() * 1000)
        _from = (
            self.from_time
            if self.from_time is not None
            else twenty_minutes_before_timestamp
        )

        body = LogsListRequest(
            filter=LogsQueryFilter(
                query=self.query,
                _from=_from,
                to=f"{self.to_time if self.to_time is not None else now_timestamp}",
            ),
            sort=LogsSort.TIMESTAMP_ASCENDING,
            page=LogsListRequestPage(
                limit=self.limit,
            ),
        )

        with ApiClient(configuration=self.configuration) as api_client:
            api_instance = LogsApi(api_client)
            response = api_instance.list_logs(body=body).to_dict()

        docs: List[Document] = []
        for row in response["data"]:
            docs.append(self.parse_log(row))

        return docs
