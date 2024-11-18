"""Lindorm AI embedding model."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)


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
    client: Any

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types to be used

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Ensure the client is initialized properly."""
        if not values.get("client"):
            try:
                import lindormai
            except ImportError:
                raise ImportError(
                    "Could not import lindormai python package. "
                    "Please install it with `pip install lindormai-x.y.z-py3-none-any.whl`."
                )

            from lindormai.model_manager import ModelManager
            values["client"] = ModelManager(values['endpoint'], values['username'], values['password'])
        return values

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for the input text.
        Args:
            text (str): Text for which embedding is to be generated.
        Returns:
            List[float]: Embedding of the input text, as a list of floating-point numbers.
        """
        response = self.client.infer(name=self.model_name, input_data=text)
        return response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts.
        Args:
            texts (List[str]): List of texts for which embeddings are to be generated.
        Returns:
            List[List[float]]: A list of embeddings for each document in the input list. Each embedding is represented as a list of floating-point numbers.
        """
        response = self.client.infer(name=self.model_name, input_data=texts)
        return response

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `embed_query`. Official does not support asynchronous requests")
