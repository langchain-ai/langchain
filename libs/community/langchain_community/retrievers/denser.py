from typing import Any, List, Optional

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class DenserRetriever(BaseRetriever):
    """Retriever that uses the DenserRetriever API for document retrieval.

    This retriever interfaces with the DenserRetriever API to fetch relevant documents
    based on a given query. It supports both synchronous and asynchronous retrieval.

    Attributes:
        api_url (str): The URL of the DenserRetriever API endpoint.
        retriever_id (str): The unique identifier for the specific retriever to use.
        top_k (Optional[int]): The number of top results to retrieve. Defaults to 10.
        api_key (Optional[str]): The API key for authentication with the API.
    """

    api_url: str
    retriever_id: str
    top_k: Optional[int]
    api_key: Optional[str]

    def _get_headers(self) -> dict:
        """Construct the headers for the API request.

        Returns:
            dict: A dictionary of headers, including Content-Type
              and optionally Authorization.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_payload(self, query: str) -> dict:
        """Construct the payload for the API request.

        Args:
            query (str): The search query.

        Returns:
            dict: A dictionary containing the query, retriever ID,
              and optionally the top_k value.
        """
        payload = {
            "query": query,
            "id": self.retriever_id,
        }
        if self.top_k is not None:
            payload["k"] = self.top_k
        return payload

    def _process_response(self, data: dict) -> List[Document]:
        """Process the API response and convert it to a list of Document objects.

        Args:
            data (dict): The JSON response from the API.

        Returns:
            List[Document]: A list of Documents containing the retrieved passages.
        """
        passages = data["passages"]
        return [
            Document(
                page_content=r["page_content"],
                metadata={"score": r["score"], **r.get("metadata", {})},
            )
            for r in passages
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve relevant documents for a given query synchronously.

        Args:
            query (str): The search query.
            run_manager (CallbackManagerForRetrieverRun):
              The callback manager for the retriever run.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of relevant Document objects.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        response = requests.post(
            self.api_url,
            json=self._get_payload(query),
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return self._process_response(response.json())

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve relevant documents for a given query asynchronously.

        Args:
            query (str): The search query.
            run_manager (AsyncCallbackManagerForRetrieverRun):
              The async callback manager for the retriever run.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of relevant Document objects.

        Raises:
            aiohttp.ClientResponseError: If the API request fails.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json=self._get_payload(query),
                headers=self._get_headers(),
            ) as response:
                response.raise_for_status()
                data = await response.json()
        return self._process_response(data)
