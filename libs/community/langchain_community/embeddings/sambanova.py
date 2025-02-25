import json
from typing import Dict, Generator, List, Optional

import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import BaseModel, ConfigDict


@deprecated(
    since="0.3.16",
    removal="1.0",
    alternative_import="langchain_sambanova.SambaStudioEmbeddings",
)
class SambaStudioEmbeddings(BaseModel, Embeddings):
    """SambaNova embedding models.

    To use, you should have the environment variables
    ``SAMBASTUDIO_EMBEDDINGS_BASE_URL``, ``SAMBASTUDIO_EMBEDDINGS_BASE_URI``
    ``SAMBASTUDIO_EMBEDDINGS_PROJECT_ID``, ``SAMBASTUDIO_EMBEDDINGS_ENDPOINT_ID``,
    ``SAMBASTUDIO_EMBEDDINGS_API_KEY``
    set with your personal sambastudio variable or pass it as a named parameter
    to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SambaStudioEmbeddings

            embeddings = SambaStudioEmbeddings(sambastudio_embeddings_base_url=base_url,
                                          sambastudio_embeddings_base_uri=base_uri,
                                          sambastudio_embeddings_project_id=project_id,
                                          sambastudio_embeddings_endpoint_id=endpoint_id,
                                          sambastudio_embeddings_api_key=api_key,
                                          batch_size=32)
            (or)

            embeddings = SambaStudioEmbeddings(batch_size=32)

            (or)

            # CoE example
            embeddings = SambaStudioEmbeddings(
                batch_size=1,
                model_kwargs={
                    'select_expert':'e5-mistral-7b-instruct'
                }
            )
    """

    sambastudio_embeddings_base_url: str = ""
    """Base url to use"""

    sambastudio_embeddings_base_uri: str = ""
    """endpoint base uri"""

    sambastudio_embeddings_project_id: str = ""
    """Project id on sambastudio for model"""

    sambastudio_embeddings_endpoint_id: str = ""
    """endpoint id on sambastudio for model"""

    sambastudio_embeddings_api_key: str = ""
    """sambastudio api key"""

    model_kwargs: dict = {}
    """Key word arguments to pass to the model."""

    batch_size: int = 32
    """Batch size for the embedding models"""

    model_config = ConfigDict(protected_namespaces=())

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["sambastudio_embeddings_base_url"] = get_from_dict_or_env(
            values, "sambastudio_embeddings_base_url", "SAMBASTUDIO_EMBEDDINGS_BASE_URL"
        )
        values["sambastudio_embeddings_base_uri"] = get_from_dict_or_env(
            values,
            "sambastudio_embeddings_base_uri",
            "SAMBASTUDIO_EMBEDDINGS_BASE_URI",
            default="api/predict/generic",
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

    def _get_tuning_params(self) -> str:
        """
        Get the tuning parameters to use when calling the model

        Returns:
            The tuning parameters as a JSON string.
        """
        if "api/v2/predict/generic" in self.sambastudio_embeddings_base_uri:
            tuning_params_dict = self.model_kwargs
        else:
            tuning_params_dict = {
                k: {"type": type(v).__name__, "value": str(v)}
                for k, v in (self.model_kwargs.items())
            }
        tuning_params = json.dumps(tuning_params_dict)
        return tuning_params

    def _get_full_url(self, path: str) -> str:
        """
        Return the full API URL for a given path.

        :param str path: the sub-path
        :returns: the full API URL for the sub-path
        :rtype: str
        """
        return f"{self.sambastudio_embeddings_base_url}/{self.sambastudio_embeddings_base_uri}/{path}"  # noqa: E501

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
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Returns a list of embeddings for the given sentences.
        Args:
            texts (`List[str]`): List of texts to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        if batch_size is None:
            batch_size = self.batch_size
        http_session = requests.Session()
        url = self._get_full_url(
            f"{self.sambastudio_embeddings_project_id}/{self.sambastudio_embeddings_endpoint_id}"
        )
        params = json.loads(self._get_tuning_params())
        embeddings = []

        if "api/predict/nlp" in self.sambastudio_embeddings_base_uri:
            for batch in self._iterate_over_batches(texts, batch_size):
                data = {"inputs": batch, "params": params}
                response = http_session.post(
                    url,
                    headers={"key": self.sambastudio_embeddings_api_key},
                    json=data,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}.\n Details: {response.text}"
                    )
                try:
                    embedding = response.json()["data"]
                    embeddings.extend(embedding)
                except KeyError:
                    raise KeyError(
                        "'data' not found in endpoint response",
                        response.json(),
                    )

        elif "api/v2/predict/generic" in self.sambastudio_embeddings_base_uri:
            for batch in self._iterate_over_batches(texts, batch_size):
                items = [
                    {"id": f"item{i}", "value": item} for i, item in enumerate(batch)
                ]
                data = {"items": items, "params": params}
                response = http_session.post(
                    url,
                    headers={"key": self.sambastudio_embeddings_api_key},
                    json=data,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}.\n Details: {response.text}"
                    )
                try:
                    embedding = [item["value"] for item in response.json()["items"]]
                    embeddings.extend(embedding)
                except KeyError:
                    raise KeyError(
                        "'items' not found in endpoint response",
                        response.json(),
                    )

        elif "api/predict/generic" in self.sambastudio_embeddings_base_uri:
            for batch in self._iterate_over_batches(texts, batch_size):
                data = {"instances": batch, "params": params}
                response = http_session.post(
                    url,
                    headers={"key": self.sambastudio_embeddings_api_key},
                    json=data,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}.\n Details: {response.text}"
                    )
                try:
                    if params.get("select_expert"):
                        embedding = response.json()["predictions"]
                    else:
                        embedding = response.json()["predictions"]
                    embeddings.extend(embedding)
                except KeyError:
                    raise KeyError(
                        "'predictions' not found in endpoint response",
                        response.json(),
                    )

        else:
            raise ValueError(
                f"handling of endpoint uri: {self.sambastudio_embeddings_base_uri} not implemented"  # noqa: E501
            )

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
        params = json.loads(self._get_tuning_params())

        if "api/predict/nlp" in self.sambastudio_embeddings_base_uri:
            data = {"inputs": [text], "params": params}
            response = http_session.post(
                url,
                headers={"key": self.sambastudio_embeddings_api_key},
                json=data,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}.\n Details: {response.text}"
                )
            try:
                embedding = response.json()["data"][0]
            except KeyError:
                raise KeyError(
                    "'data' not found in endpoint response",
                    response.json(),
                )

        elif "api/v2/predict/generic" in self.sambastudio_embeddings_base_uri:
            data = {"items": [{"id": "item0", "value": text}], "params": params}
            response = http_session.post(
                url,
                headers={"key": self.sambastudio_embeddings_api_key},
                json=data,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}.\n Details: {response.text}"
                )
            try:
                embedding = response.json()["items"][0]["value"]
            except KeyError:
                raise KeyError(
                    "'items' not found in endpoint response",
                    response.json(),
                )

        elif "api/predict/generic" in self.sambastudio_embeddings_base_uri:
            data = {"instances": [text], "params": params}
            response = http_session.post(
                url,
                headers={"key": self.sambastudio_embeddings_api_key},
                json=data,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}.\n Details: {response.text}"
                )
            try:
                if params.get("select_expert"):
                    embedding = response.json()["predictions"][0]
                else:
                    embedding = response.json()["predictions"][0]
            except KeyError:
                raise KeyError(
                    "'predictions' not found in endpoint response",
                    response.json(),
                )

        else:
            raise ValueError(
                f"handling of endpoint uri: {self.sambastudio_embeddings_base_uri} not implemented"  # noqa: E501
            )

        return embedding
