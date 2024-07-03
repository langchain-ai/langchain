from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import get_from_dict_or_env, pre_init


class NLPCloudEmbeddings(BaseModel, Embeddings):
    """NLP Cloud embedding models.

    To use, you should have the nlpcloud python package installed

    Example:
        .. code-block:: python

            from langchain_community.embeddings import NLPCloudEmbeddings

            embeddings = NLPCloudEmbeddings()
    """

    model_name: str  # Define model_name as a class attribute
    gpu: bool  # Define gpu as a class attribute
    client: Any  #: :meta private:

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, gpu=gpu, **kwargs)

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        nlpcloud_api_key = get_from_dict_or_env(
            values, "nlpcloud_api_key", "NLPCLOUD_API_KEY"
        )
        try:
            import nlpcloud

            values["client"] = nlpcloud.Client(
                values["model_name"], nlpcloud_api_key, gpu=values["gpu"], lang="en"
            )
        except ImportError:
            raise ImportError(
                "Could not import nlpcloud python package. "
                "Please install it with `pip install nlpcloud`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using NLP Cloud.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        return self.client.embeddings(texts)["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using NLP Cloud.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.client.embeddings([text])["embeddings"][0]
