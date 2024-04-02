from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator


def is_endpoint_live(url: str, headers: Optional[dict], payload: Any) -> bool:
    """
    Check if an endpoint is live by sending a GET request to the specified URL.

    Args:
        url (str): The URL of the endpoint to check.

    Returns:
        bool: True if the endpoint is live (status code 200), False otherwise.

    Raises:
        Exception: If the endpoint returns a non-successful status code or if there is
            an error querying the endpoint.
    """
    try:
        response = requests.request("POST", url, headers=headers, data=payload)

        # Check if the status code is 200 (OK)
        if response.status_code == 200:
            return True
        else:
            # Raise an exception if the status code is not 200
            raise Exception(
                f"Endpoint returned a non-successful status code: "
                f"{response.status_code}"
            )
    except requests.exceptions.RequestException as e:
        # Handle any exceptions (e.g., connection errors)
        raise Exception(f"Error querying the endpoint: {e}")


class NeMoEmbeddings(BaseModel, Embeddings):
    batch_size: int = 16
    model: str = "NV-Embed-QA-003"
    api_endpoint_url: str = "http://localhost:8088/v1/embeddings"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the end point is alive using the values that are provided."""

        url = values["api_endpoint_url"]
        model = values["model"]

        # Optional: A minimal test payload and headers required by the endpoint
        headers = {"Content-Type": "application/json"}
        payload = json.dumps(
            {"input": "Hello World", "model": model, "input_type": "query"}
        )

        is_endpoint_live(url, headers, payload)

        return values

    async def _aembedding_func(
        self, session: Any, text: str, input_type: str
    ) -> List[float]:
        """Async call out to embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        headers = {"Content-Type": "application/json"}

        async with session.post(
            self.api_endpoint_url,
            json={"input": text, "model": self.model, "input_type": input_type},
            headers=headers,
        ) as response:
            response.raise_for_status()
            answer = await response.text()
            answer = json.loads(answer)
            return answer["data"][0]["embedding"]

    def _embedding_func(self, text: str, input_type: str) -> List[float]:
        """Call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        payload = json.dumps(
            {"input": text, "model": self.model, "input_type": input_type}
        )
        headers = {"Content-Type": "application/json"}

        response = requests.request(
            "POST", self.api_endpoint_url, headers=headers, data=payload
        )
        response_json = json.loads(response.text)
        embedding = response_json["data"][0]["embedding"]

        return embedding

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return [self._embedding_func(text, input_type="passage") for text in documents]

    def embed_query(self, text: str) -> List[float]:
        return self._embedding_func(text, input_type="query")

    async def aembed_query(self, text: str) -> List[float]:
        """Call out to NeMo's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """

        async with aiohttp.ClientSession() as session:
            embedding = await self._aembedding_func(session, text, "passage")
            return embedding

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to NeMo's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []

        async with aiohttp.ClientSession() as session:
            for batch in range(0, len(texts), self.batch_size):
                text_batch = texts[batch : batch + self.batch_size]

                for text in text_batch:
                    # Create tasks for all texts in the batch
                    tasks = [
                        self._aembedding_func(session, text, "passage")
                        for text in text_batch
                    ]

                    # Run all tasks concurrently
                    batch_results = await asyncio.gather(*tasks)

                    # Extend the embeddings list with results from this batch
                    embeddings.extend(batch_results)

        return embeddings
