from typing import Dict, Generator, List

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env


class SambaStudioEmbeddings(BaseModel, Embeddings):
    """SambaNova embedding models.

    To use, you should have the environment variables
    ``SAMBASTUDIO_EMBEDDINGS_BASE_URL``, ``SAMBASTUDIO_EMBEDDINGS_PROJECT_ID``,
    ``SAMBASTUDIO_EMBEDDINGS_ENDPOINT_ID``, ``SAMBASTUDIO_EMBEDDINGS_API_KEY``,
    set with your personal sambastudio variable or pass it as a named parameter
    to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SambaStudioEmbeddings
            embeddings = SambaStudioEmbeddings(sambastudio_embeddings_base_url=base_url,
                                          sambastudio_embeddings_project_id=project_id,
                                          sambastudio_embeddings_endpoint_id=endpoint_id,
                                          sambastudio_embeddings_api_key=api_key)
                             (or)
            embeddings = SambaStudioEmbeddings()
    """

    API_BASE_PATH = "/api/predict/nlp/"
    """Base path to use for the API usage"""

    sambastudio_embeddings_base_url: str = ""
    """Base url to use"""

    sambastudio_embeddings_project_id: str = ""
    """Project id on sambastudio for model"""

    sambastudio_embeddings_endpoint_id: str = ""
    """endpoint id on sambastudio for model"""

    sambastudio_embeddings_api_key: str = ""
    """sambastudio api key"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["sambastudio_embeddings_base_url"] = get_from_dict_or_env(
            values, "sambastudio_embeddings_base_url", "SAMBASTUDIO_EMBEDDINGS_BASE_URL"
        )
        values["sambastudio_embeddings_project_id"] = get_from_dict_or_env(
            values,
            "sambastudio_embeddings_project_id",
            "SAMBASTUDIO_EMBEDDINGS_PROJECT_ID",
        )
        values["sambastudio_embeddings_endpoint_id"] = get_from_dict_or_env(
            values,
            "sambastudio_embeddings_endpoint_id",
            "SAMBASTUDIO_EMBEDDINGS_ENDPOINT_ID",
        )
        values["sambastudio_embeddings_api_key"] = get_from_dict_or_env(
            values, "sambastudio_embeddings_api_key", "SAMBASTUDIO_EMBEDDINGS_API_KEY"
        )
        return values

    def _get_full_url(self, path: str) -> str:
        """
        Return the full API URL for a given path.

        :param str path: the sub-path
        :returns: the full API URL for the sub-path
        :rtype: str
        """
        return f"{self.sambastudio_embeddings_base_url}{self.API_BASE_PATH}{path}"

    def _iterate_over_batches(self, texts: List[str], batch_size: int) -> Generator:
        """Generator for creating batches in the embed documents method
        Args:
            texts (List[str]): list of strings to embed
            batch_size (int, optional): batch size to be used for the embedding model.
            Will depend on the RDU endpoint used.
        Yields:
            List[str]: list (batch) of strings of size batch size
        """
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def embed_documents(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Returns a list of embeddings for the given sentences.
        Args:
            texts (`List[str]`): List of texts to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        http_session = requests.Session()
        url = self._get_full_url(
            f"{self.sambastudio_embeddings_project_id}/{self.sambastudio_embeddings_endpoint_id}"
        )

        embeddings = []

        for batch in self._iterate_over_batches(texts, batch_size):
            data = {"inputs": batch}
            response = http_session.post(
                url,
                headers={"key": self.sambastudio_embeddings_api_key},
                json=data,
            )
            embedding = response.json()["data"]
            embeddings.extend(embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        http_session = requests.Session()
        url = self._get_full_url(
            f"{self.sambastudio_embeddings_project_id}/{self.sambastudio_embeddings_endpoint_id}"
        )

        data = {"inputs": [text]}

        response = http_session.post(
            url,
            headers={"key": self.sambastudio_embeddings_api_key},
            json=data,
        )
        embedding = response.json()["data"][0]

        return embedding
