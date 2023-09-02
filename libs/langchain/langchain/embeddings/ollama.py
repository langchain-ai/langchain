import requests
from langchain.llms.ollama import _OllamaCommon
from typing import Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, root_validator


class OllamaEmbeddings(BaseModel, Embeddings, _OllamaCommon):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Example:
        .. code-block:: python

            from langchain.embeddings import OllamaEmbeddings
            ollama_emb = OllamaEmbeddings(
                model="llama:7b",
            )
            r1 = ollama_emb.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            r2 = ollama_emb.embed_query(
                "What is the second letter of Greek alphabet"
            )

    """

    model: str = super().model
    """Embeddings model to use."""
    embed_instruction: str = "passage: "
    """Instruction used to embed documents."""
    query_instruction: str = "query: "
    """Instruction used to embed the query."""
    model_kwargs: Optional[dict] = None
    """Other model keyword args"""

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        return values
    

    def _process_emb_response(self, input: str) -> List[float]:
        """Process a response from the API.

        Args:
            response: The response from the API.

        Returns:
            The response as a dictionary.
        """
        headers = {
            "Content-Type": "application/json",
        }

        try:
            res = requests.post(
                f"{_OllamaCommon.base_url}/api/embeddings",
                headers=headers,
                json={
                    "model": self.model,
                    "prompt": input,
                },
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (res.status_code, res.text)
            )
        try:
            t = res.json()
            return t["embeddings"]
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )

    def _embed(self, input: List[str]) -> List[List[float]]:
        embeddings_list = List[List[float]]
        for prompt in input:
            embeddings = self._process_emb_response(prompt)
            embeddings_list.append(embeddings)

        return embeddings_list

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a Deep Infra deployed embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [f"{self.query_instruction}{text}" for text in texts]
        embeddings = self._embed(instruction_pairs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Deep Infra deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = f"{self.query_instruction}{text}"
        embedding = self._embed([instruction_pair])[0]
        return embedding
