"""Wrapper around OpenAI embedding models."""
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings


class OpenAIEmbeddings(BaseModel, Embeddings):
    """Wrapper around OpenAI embedding models."""

    client: Any  #: :meta private:
    model_name: str = "babbage"
    """Model name to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "Did not find OpenAI API key, please add an environment variable"
                " `OPENAI_API_KEY` which contains it."
            )
        try:
            import openai

            values["client"] = openai.Embedding
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    def embedding_func(self, text: str, *, engine: str) -> List[float]:
        """"""
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return self.client.create(input=[text], engine=engine)["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs."""
        responses = [
            self.embedding_func(text, engine=f"text-search-{self.model_name}-doc-001")
            for text in texts
        ]
        return responses

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text."""
        embedding = self.embedding_func(
            text, engine=f"text-search-{self.model_name}-query-001"
        )
        return embedding
