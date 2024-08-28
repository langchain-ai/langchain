from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """ZhipuAI embedding model integration.

    Setup:

        To use, you should have the ``zhipuai`` python package installed, and the
        environment variable ``ZHIPU_API_KEY`` set with your API KEY.

        More instructions about ZhipuAi Embeddings, you can get it
        from  https://open.bigmodel.cn/dev/api#vector

        .. code-block:: bash

            pip install -U zhipuai
            export ZHIPU_API_KEY="your-api-key"

    Key init args — completion params:
        model: Optional[str]
            Name of ZhipuAI model to use.
        api_key: str
            Automatically inferred from env var `ZHIPU_API_KEY` if not provided.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:

        .. code-block:: python

            from langchain_community.embeddings import ZhipuAIEmbeddings

            embed = ZhipuAIEmbeddings(
                model="embedding-2",
                # api_key="...",
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python

            [-0.003832892, 0.049372625, -0.035413884, -0.019301128, 0.0068899863, 0.01248398, -0.022153955, 0.006623926, 0.00778216, 0.009558191, ...]


    Embed multiple text:
        .. code-block:: python

            input_texts = ["This is a test query1.", "This is a test query2."]
            embed.embed_documents(input_texts)

        .. code-block:: python

            [
                [0.0083934665, 0.037985895, -0.06684559, -0.039616987, 0.015481004, -0.023952313, ...],
                [-0.02713102, -0.005470169, 0.032321047, 0.042484466, 0.023290444, 0.02170547, ...]
            ]
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = Field(default="embedding-2")
    """Model name"""
    api_key: str
    """Automatically inferred from env var `ZHIPU_API_KEY` if not provided."""
    dimensions: Optional[int] = None
    """The number of dimensions the resulting output embeddings should have.

    Only supported in `embedding-3` and later models.
    """

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        values["api_key"] = get_from_dict_or_env(values, "api_key", "ZHIPUAI_API_KEY")
        try:
            from zhipuai import ZhipuAI

            values["client"] = ZhipuAI(api_key=values["api_key"])
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
        if self.dimensions is not None:
            resp = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )
        else:
            resp = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = [r.embedding for r in resp.data]
        return embeddings
