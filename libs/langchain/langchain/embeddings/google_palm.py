from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator() -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle PaLM exceptions"""
    import google.api_core.exceptions

    multiplier = 2
    min_seconds = 1
    max_seconds = 60
    max_retries = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=multiplier, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
            | retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
            | retry_if_exception_type(google.api_core.exceptions.GoogleAPIError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(
    embeddings: GooglePalmEmbeddings, *args: Any, **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _embed_with_retry(*args: Any, **kwargs: Any) -> Any:
        return embeddings.client.generate_embeddings(*args, **kwargs)

    return _embed_with_retry(*args, **kwargs)


class GooglePalmEmbeddings(BaseModel, Embeddings):
    """Google's PaLM Embeddings APIs."""

    client: Any
    google_api_key: Optional[str]
    model_name: str = "models/embedding-gecko-001"
    """Model name to use."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
        except ImportError:
            raise ImportError("Could not import google.generativeai python package.")

        values["client"] = genai

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = embed_with_retry(self, self.model_name, text)
        return embedding["embedding"]
