"""Wrapper around Xinference embedding models."""
from typing import Any, List, Optional

from langchain.embeddings.base import Embeddings

class XinferenceEmbeddings(Embeddings):

    """Wrapper around xinference embedding models.
    To use, you should have the xinference library installed:
    .. code-block:: bash
        pip install xinference
    Check out: https://github.com/xorbitsai/inference
    To run, you need to start a Xinference supervisor on one server and Xinference workers on the other servers
    Example:
        Starting the supervisor:
        .. code-block:: bash
            $ xinference-supervisor
        Starting the worker:
        .. code-block:: bash
            $ xinference-worker

    
    To use Xinference with LangChain, you need to first launch a model. You can use the RESTfulClient to do so:
    
    Example:
    .. code-block:: python
        from xinference.client import RESTfulClient
        client = RESTfulClient("http://0.0.0.0:9997")
        model_uid = client.launch_model(model_name="orca", quantization="q4_0", embedding="True")
    
    Then you can use Xinference Embedding with LangChain.

    Example:
    .. code-block:: python
        from langchain.embeddings import XinferenceEmbeddings

        xinference = XinferenceEmbeddings(
            server_url="http://0.0.0.0:9997",
            model_uid = model_uid
        )

    """

    client: Any
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_uid: Optional[str] = None
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError as e:
            raise ImportError(
                "Could not import RESTfulClient from xinference. Make sure to install xinference in advance"
            ) from e

        super().__init__()

        if server_url is None:
            raise ValueError(f"Please provide server URL")

        if model_uid is None:
            raise ValueError(f"Please provide the model UID")

        self.server_url = server_url

        self.model_uid = model_uid

        self.client = RESTfulClient(server_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Xinference.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """

        model = self.client.get_model(self.model_uid)

        embeddings = [
            model.create_embedding(text)["data"][0]["embedding"] for text in texts
        ]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query of documents using Xinference.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """

        model = self.client.get_model(self.model_uid)

        embedding_res = model.create_embedding(text)

        embedding = embedding_res["data"][0]["embedding"]

        return list(map(float, embedding))