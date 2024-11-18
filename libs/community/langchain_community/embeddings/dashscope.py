from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = {"text-embedding-v1": 25, "text-embedding-v2": 25, "text-embedding-v3": 6}


def _create_retry_decorator(embeddings: DashScopeEmbeddings) -> Callable[[Any], Any]:
    multiplier = 1
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 1 seconds, then up to 4 seconds, then 4 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(embeddings: DashScopeEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        result = []
        i = 0
        input_data = kwargs["input"]
        input_len = len(input_data) if isinstance(input_data, list) else 1
        batch_size = BATCH_SIZE.get(kwargs["model"], 25)
        while i < input_len:
            kwargs["input"] = (
                input_data[i : i + batch_size]
                if isinstance(input_data, list)
                else input_data
            )
            resp = embeddings.client.call(**kwargs)
            if resp.status_code == 200:
                result += resp.output["embeddings"]
            elif resp.status_code in [400, 401]:
                raise ValueError(
                    f"status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}"
                )
            else:
                raise HTTPError(
                    f"HTTP error occurred: status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}",
                    response=resp,
                )
            i += batch_size
        return result

    return _embed_with_retry(**kwargs)


class DashScopeEmbeddings(BaseModel, Embeddings):
    """Tongyi embedding model integration.

    Setup:
        Install ``dashscope`` python package and set environment variable ``DASHSCOPE_API_KEY``.

    Key init args — embedding params:
        model: str
            Name of Tongyi model to use.

    Key init args — client params:
        api_key: Optional[str] = None
            Tongyi API key.
        max_retries: int = 5
            Maximum number of retries to make when generating.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.embeddings import DashScopeEmbeddings

            embeddings = DashScopeEmbeddings(model='text-embedding-v2')

    Embed single text:
        .. code-block:: python

            input_text = "塞翁失马，焉知非福"
            vector = embeddings.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [1.514211893081665, 2.2103159427642822, 3.0578529834747314]

    Embed multiple texts:
        .. code-block:: python

            input_texts = ['风急天高猿啸哀', '渚清沙白鸟飞回', '无边落木萧萧下', '不尽长江滚滚来']
            vectors = embeddings.embed_documents(input_texts)
            # Showing only the first 3 coordinates
            print(len(vectors))
            print(vectors[0][:3])

        .. code-block:: python

            4
            [1.5536729097366333, -2.237586736679077, 1.5397623777389526]

    Async:
        .. code-block:: python

            vector = await embeddings.aembed_query(input_text)
            print(vector[:3])

            # multiple:
            # await embeddings.aembed_documents(input_texts)

        .. code-block:: python

            [1.514211893081665, 2.2103159427642822, 3.0578529834747314]

    """  # noqa: E501

    client: Any = None  #: :meta private:
    """The DashScope client."""
    model: str = "text-embedding-v1"
    dashscope_api_key: Optional[str] = Field(None, alias="api_key")
    max_retries: int = 5
    """Maximum number of retries to make when generating."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        import dashscope

        """Validate that api key and python package exists in environment."""
        values["dashscope_api_key"] = get_from_dict_or_env(
            values, ["dashscope_api_key", "api_key"], "DASHSCOPE_API_KEY"
        )
        dashscope.api_key = values["dashscope_api_key"]
        try:
            import dashscope

            values["client"] = dashscope.TextEmbedding
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to DashScope's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = embed_with_retry(
            self, input=texts, text_type="document", model=self.model
        )
        embedding_list = [item["embedding"] for item in embeddings]
        return embedding_list

    def embed_query(self, text: str) -> List[float]:
        """Call out to DashScope's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = embed_with_retry(
            self, input=text, text_type="query", model=self.model
        )[0]["embedding"]
        return embedding
