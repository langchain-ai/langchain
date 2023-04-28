"""Wrapper arround Google's PaLM Embeddings APIs."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env


class GooglePalmEmbeddings(BaseModel, Embeddings):
    client: Any
    google_api_key: Optional[str]
    model_name: str = "models/embedding-gecko-001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
        except ImportError:
            raise ImportError("Could not import google.generativeai python package.")

        values["client"] = genai

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.client.generate_embeddings(self.model_name, text)
        return embedding["embedding"]
