"""Lindorm AI embedding model."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import requests
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

HTTP_HDR_AK_KEY = "x-ld-ak"
HTTP_HDR_SK_KEY = "x-ld-sk"
REST_URL_PATH = "/v1/ai"
REST_URL_MODELS_PATH = REST_URL_PATH + "/models"
INFER_INPUT_KEY = "input"
INFER_PARAMS_KEY = "params"
RSP_DATA_KEY = "data"
RSP_MODELS_KEY = "models"


class LindormAIEmbeddings(BaseModel, Embeddings):
    """`LindormAI embedding models API.

    To use, you should have the ``lindormai`` python package installed,
    and set variable ``endpoint``, ``username``, ``password`` and ``model_name``.

    Example:
        .. code-block:: python
            from langchain_community.embeddings.lindormai import LindormAIEmbeddings
            lindorm_ai_embedding = LindormAIEmbeddings(
                endpoint='https://ld-xxx-proxy-ml.lindorm.rds.aliyuncs.com:9002',
                username='root',
                password='xxx',
                model_name='bge_model'
            )
    """

    endpoint: str = Field(
        ...,
        description="The endpoint of Lindorm AI to use.",
    )
    username: str = Field(
        ...,
        description="Lindorm username.",
    )
    password: str = Field(
        ...,
        description="Lindorm password.",
    )
    model_name: str = Field(
        ...,
        description="The model to use.",
    )
    client: Any = Field(None, description="Created Opensearch Client")

    class Config:
        protected_namespaces = ()
        arbitrary_types_allowed = True  # Allow arbitrary types to be used

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Ensure the client is initialized properly."""
        if not values.get("client"):
            if values.get("username") is None:
                raise ValueError("username can't be empty")
            if values.get("password") is None:
                raise ValueError("password can't be empty")
            if values.get("endpoint") is None:
                raise ValueError("endpoint can't be empty")
        return values

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4))
    def __post_with_retry(
        self, url: str, data: Any = None, json: Any = None, **kwargs: Any
    ) -> Any:
        response = requests.post(url=url, data=data, json=json, **kwargs)
        response.raise_for_status()
        return response

    def _infer(self, model_name: str, input_data: Any, params: Any) -> Any:
        url = f"{self.endpoint}{REST_URL_MODELS_PATH}/{model_name}/infer"
        infer_dict = {INFER_INPUT_KEY: input_data, INFER_PARAMS_KEY: params}
        result = None
        try:
            headers = {HTTP_HDR_AK_KEY: self.username, HTTP_HDR_SK_KEY: self.password}
            response = self.__post_with_retry(url, json=infer_dict, headers=headers)
            response.raise_for_status()
            result = response.json()
        except Exception as error:
            logger.error(
                f"infer model for {model_name} with "
                f"input {input_data} and params {params}: {error}"
            )
        return result[RSP_DATA_KEY] if result else None

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for the input text.
        Args:
            text (str): Text for which embedding is to be generated.
        Returns:
            List[float]: Embedding of the input text, as a list of floating-point
                        numbers.
        """
        response = self._infer(model_name=self.model_name, input_data=text, params={})
        return response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts.
        Args:
            texts (List[str]): List of texts for which embeddings are to be generated.
        Returns:
            List[List[float]]: A list of embeddings for each document in the input list.
                Each embedding is represented as a list of floating-point numbers.
        """
        response = self._infer(model_name=self.model_name, input_data=texts, params={})
        return response

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError(
            "Please use `embed_documents`. "
            "Official does not support asynchronous requests"
        )

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError(
            "Please use `embed_query`. Official does not support asynchronous requests"
        )
