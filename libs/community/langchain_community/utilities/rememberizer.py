"""Wrapper for Rememberizer APIs."""

from typing import Any, Dict, List, Optional, cast

import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, model_validator


class RememberizerAPIWrapper(BaseModel):
    """Wrapper for Rememberizer APIs."""

    top_k_results: int = 10
    rememberizer_api_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key in environment."""
        rememberizer_api_key = get_from_dict_or_env(
            values, "rememberizer_api_key", "REMEMBERIZER_API_KEY"
        )
        values["rememberizer_api_key"] = rememberizer_api_key

        return values

    def search(self, query: str) -> dict:
        """Search for a query in the Rememberizer API."""
        url = f"https://api.rememberizer.ai/api/v1/documents/search?q={query}&n={self.top_k_results}"
        response = requests.get(
            url, headers={"x-api-key": cast(str, self.rememberizer_api_key)}
        )
        data = response.json()

        if response.status_code != 200:
            raise ValueError(f"API Error: {data}")

        matched_chunks = data.get("matched_chunks", [])
        return matched_chunks

    def load(self, query: str) -> List[Document]:
        matched_chunks = self.search(query)
        docs = []
        for matched_chunk in matched_chunks:
            docs.append(
                Document(
                    page_content=matched_chunk["matched_content"],
                    metadata=matched_chunk["document"],
                )
            )
        return docs
