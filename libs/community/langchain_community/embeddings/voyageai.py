from __future__ import annotations

import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: VoyageEmbeddings) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _check_response(response: dict) -> dict:
    if "data" not in response:
        raise RuntimeError(f"Voyage API Error. Message: {json.dumps(response)}")
    return response


def embed_with_retry(embeddings: VoyageEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        response = requests.post(**kwargs)
        return _check_response(response.json())

    return _embed_with_retry(**kwargs)


class VoyageEmbeddings(BaseModel, Embeddings):
    """Voyage embedding models.

    To use, you should have the environment variable ``VOYAGE_API_KEY`` set with
    your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import VoyageEmbeddings

            voyage = VoyageEmbeddings(voyage_api_key="your-api-key")
            text = "This is a test query."
            query_result = voyage.embed_query(text)
    """

    model: str = "voyage-01"
    voyage_api_base: str = "https://api.voyageai.com/v1/embeddings"
    voyage_api_key: Optional[SecretStr] = None
    batch_size: int = 8
    """Maximum number of texts to embed in each API request."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout in seconds for the API request."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding. Must have tqdm installed if set 
        to True."""
    truncation: Optional[bool] = None
    """Whether to truncate the input texts to fit within the context length.
    
        If True, over-length input texts will be truncated to fit within the context 
        length, before vectorized by the embedding model. If False, an error will be 
        raised if any given text exceeds the context length. If not specified 
        (defaults to None), we will truncate the input text before sending it to the 
        embedding model if it slightly exceeds the context window length. If it 
        significantly exceeds the context window length, an error will be raised."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["voyage_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "voyage_api_key", "VOYAGE_API_KEY")
        )
        return values

    def _invocation_params(
        self, input: List[str], input_type: Optional[str] = None
    ) -> Dict:
        api_key = cast(SecretStr, self.voyage_api_key).get_secret_value()
        params: Dict = {
            "url": self.voyage_api_base,
            "headers": {"Authorization": f"Bearer {api_key}"},
            "json": {"model": self.model, "input": input, "input_type": input_type},
            "timeout": self.request_timeout,
        }
        if self.truncation is not None:
            params["json"]["truncation"] = self.truncation
        return params

    def _get_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        input_type: Optional[str] = None,
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []

        if batch_size is None:
            batch_size = self.batch_size

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            _iter = tqdm(range(0, len(texts), batch_size))
        else:
            _iter = range(0, len(texts), batch_size)

        if input_type and input_type not in ["query", "document"]:
            raise ValueError(
                f"input_type {input_type} is invalid. Options: None, 'query', "
                "'document'."
            )

        for i in _iter:
            response = embed_with_retry(
                self,
                **self._invocation_params(
                    input=texts[i : i + batch_size], input_type=input_type
                ),
            )
            embeddings.extend(r["embedding"] for r in response["data"])

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Voyage Embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self._get_embeddings(
            texts, batch_size=self.batch_size, input_type="document"
        )

    def embed_query(self, text: str) -> List[float]:
        """Call out to Voyage Embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self._get_embeddings([text], input_type="query")[0]

    def embed_general_texts(
        self, texts: List[str], *, input_type: Optional[str] = None
    ) -> List[List[float]]:
        """Call out to Voyage Embedding endpoint for embedding general text.

        Args:
            texts: The list of texts to embed.
            input_type: Type of the input text. Default to None, meaning the type is
                unspecified. Other options: query, document.

        Returns:
            Embedding for the text.
        """
        return self._get_embeddings(
            texts, batch_size=self.batch_size, input_type=input_type
        )
