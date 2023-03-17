"""Wrapper around Jina embedding models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env


class JinaEmbeddings(BaseModel, Embeddings):
    client: Any  #: :meta private:

    """Model name to use."""
    model_name: str = "ViT-B-32"

    jina_auth_token: Optional[str] = None

    @root_validator(pre=True)
    def get_endpoint(cls, values: Dict) -> Dict:
        """Get the endpoint from the model name."""

        # TODO: validate model name

        values["endpoint"] = f"https://api.jina.ai/model/{values['model_name']}"
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token and python package exists in environment."""
        jina_auth_token = get_from_dict_or_env(
            values, "jina_auth_token", "JIAN_AUTH_TOKEN"
        )
        try:
            import jina

            values["client"] = jina.Client(
                host=values["endpoint"], token=jina_auth_token
            )
        except ImportError:
            raise ValueError(
                "Could not import `jina` python package. "
                "Please it install it with `pip install jina`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Jina's embedding endpoint.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.client.embed(
            model=self.model, texts=texts, truncate=self.truncate
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Call out to Jina's embedding endpoint.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(
            model=self.model, texts=[text], truncate=self.truncate
        ).embeddings[0]
        return list(map(float, embedding))
