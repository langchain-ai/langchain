import json
from typing import List

import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field


class PerplexityWrapper(BaseModel):
    """Wrapper around the Perplexity API for chat completions using the Sonar models."""

    api_key: str = Field(..., description="The API key to use for the Perplexity API.")
    model: str = Field(
        default="sonar", description="The default model to use for chat completions."
    )
    base_url: str = Field(
        default="https://api.perplexity.ai/chat/completions",
        description="The base URL for the Perplexity API.",
    )
    search_kwargs: dict = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the API call.",
    )

    def run(self, query: str) -> str:
        """
        Query the Perplexity API and return the response as a JSON string.

        Args:
            query: The query to send for a chat completion.

        Returns:
            The API response as a JSON string.
        """
        return self.chat_completion(query=query)

    def download_documents(self, query: str) -> List[Document]:
        """
        Query the Perplexity API and return the result as a list of Documents.

        Args:
            query: The query to send for a chat completion.

        Returns:
            A list of Documents containing the API response.
        """
        response_text = self.chat_completion(query=query)
        # Wrap the raw response in a Document.
        doc = Document(page_content=response_text, metadata={"query": query})
        return [doc]

    def chat_completion(
        self,
        query: str,
        model: str = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        search_domain_filter: list = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        search_recency_filter: str = "",
        top_k: int = 0,
        stream: bool = False,
        presence_penalty: float = 0,
        frequency_penalty: float = 1,
        response_format: dict = None,
        web_search_options: dict = None,
    ) -> str:
        """
        Query the Perplexity API for a chat completion.

        Args:
            query: The user query to send to the API.
            model: The model name to use (overrides default if provided).
            max_tokens: Maximum tokens for the response.
            temperature: Randomness level.
            top_p: Nucleus sampling threshold.
            search_domain_filter: List of domains to restrict search.
            return_images: Whether to include images.
            return_related_questions: Whether to include related questions.
            search_recency_filter: Time filter for search.
            top_k: Top-k filtering parameter.
            stream: Whether to stream the response.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.
            response_format: JSON output formatting options.
            web_search_options: Options for web search.

        Returns:
            The API response as a string.
        """

        messages = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "search_domain_filter": search_domain_filter,
            "return_images": return_images,
            "return_related_questions": return_related_questions,
            "search_recency_filter": search_recency_filter,
            "top_k": top_k,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "response_format": response_format,
            "web_search_options": web_search_options,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.request(
            "POST", self.base_url, json=payload, headers=headers
        )
        if not response.ok:
            raise Exception(
                f"Perplexity API error {response.status_code}: {response.text}"
            )
        return response.text
