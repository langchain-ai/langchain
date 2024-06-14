from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """ZhipuAI embedding models.

    To use, you should have the ``zhipuai`` python package installed, and the
    environment variable ``ZHIPU_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    More instructions about ZhipuAi Embeddings, you can get it
    from  https://open.bigmodel.cn/dev/api#vector

    Example:
        .. code-block:: python

            from langchain_community.embeddings import ZhipuAIEmbeddings
            embeddings = ZhipuAIEmbeddings(api_key="your-api-key")
            text = "This is a test query."
            query_result = embeddings.embed_query(text)
            # texts = ["This is a test query1.", "This is a test query2."]
            # query_result = embeddings.embed_query(texts)
    """

    _client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = Field(default="embedding-2")
    """Model name"""
    api_key: str
    """Automatically inferred from env var `ZHIPU_API_KEY` if not provided."""

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        values["api_key"] = get_from_dict_or_env(values, "api_key", "ZHIPUAI_API_KEY")
        try:
            from zhipuai import ZhipuAI

            values["_client"] = ZhipuAI(api_key=values["api_key"])
        except ImportError:
            raise ImportError(
                "Could not import zhipuai python package."
                "Please install it with `pip install zhipuai`."
            )
        return values

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a text using the AutoVOT algorithm.

        Args:
            text: A text to embed.

        Returns:
            Input document's embedded list.
        """
        resp = self.embed_documents([text])
        return resp[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents using the AutoVOT algorithm.

        Args:
            texts: A list of text documents to embed.

        Returns:
            A list of embeddings for each document in the input list.
            Each embedding is represented as a list of float values.
        """
        resp = self._client.embeddings.create(model=self.model, input=texts)
        embeddings = [r.embedding for r in resp.data]
        return embeddings
