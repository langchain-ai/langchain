import logging
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)


class PredictionGuardEmbeddings(BaseModel, Embeddings):
    """Prediction Guard chat models.

    To use, you should have the ``predictionguard`` python package installed, and the
    environment variable ``PREDICTIONGUARD_API_KEY`` set with your api_key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            embeddings = PredictionGuardEmbeddings(
                                    model="bridgetower-large-itm-mlm-itc",
                                    api_key="my-api-key"
                                    )
    """

    client: Any = None  #: :meta private:
    """Prediction Guard Client"""

    model: Optional[str] = "bridgetower-large-itm-mlm-itc"
    """Model name to use."""

    predictionguard_api_key: Optional[str] = None
    """Prediction Guard API key."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the api_key and python package exists in environment."""
        pg_api_key = get_from_dict_or_env(
            values, "predictionguard_api_key", "PREDICTIONGUARD_API_KEY"
        )

        try:
            from predictionguard import PredictionGuard

            values["client"] = PredictionGuard(
                api_key=pg_api_key,
            )

        except ImportError:
            raise ImportError(
                "Could not import predictionguard python package. "
                "Please install it with `pip install predictionguard`."
            )

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Prediction Guard's embedding endpoint for embedding documents.

        Args:
            texts:
                List of dictionaries containing text inputs.

        Returns:
            Embeddings for the texts.
        """

        inputs = []
        for text in texts:
            input = {"text": text}
            inputs.append(input)

        response = self.client.embeddings.create(model=self.model, input=inputs)

        res = []
        indx = 0
        for re in response["data"]:
            if re["index"] == indx:
                res.append(re["embedding"])
                indx += 1
            else:
                continue

        return res

    def embed_query(self, text: str) -> List[float]:
        """Call out to Prediction Guard's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        inputs = [{"text": text}]
        response = self.client.embeddings.create(model=self.model, input=inputs)

        return response["data"][0]["embedding"]

    def embed_images(self, images: List[str]) -> List[float]:
        """
        Call out to Prediction Guard's embedding endpoint for embedding multiple images.

        Args:
            images: A list of images to embed.
            Supports image file paths, image URLs, data URIs, and base64 encoded images.

        Returns:
            Embeddings for the images.
        """

        inputs = []
        for image in images:
            input = {"image": image}

            inputs.append(input)

        response = self.client.embeddings.create(model=self.model, input=inputs)

        res = []
        indx = 0
        for re in response["data"]:
            if re["index"] == indx:
                res.append(re["embedding"])
                indx += 1
            else:
                continue

        return res

    def embed_image_text(self, inputs: List[Dict[str, str]]) -> List[float]:
        """
        Call out to Prediction Guard embeddings for embedding an image and text.

        Args:
            inputs: A list of dictionaries containing the text and images to embed.

        Returns:
            Embeddings for the text and images.
        """

        response = self.client.embeddings.create(model=self.model, input=inputs)

        res = []
        indx = 0
        for re in response["data"]:
            if re["index"] == indx:
                res.append(re["embedding"])
                indx += 1
            else:
                continue

        return res
