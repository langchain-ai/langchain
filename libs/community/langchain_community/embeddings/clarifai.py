import logging
from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ClarifaiEmbeddings(BaseModel, Embeddings):
    """Clarifai embedding models.

    To use, you should have the ``clarifai`` python package installed, and the
    environment variable ``CLARIFAI_PAT`` set with your personal access token or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import ClarifaiEmbeddings
            clarifai = ClarifaiEmbeddings(user_id=USER_ID,
                                          app_id=APP_ID,
                                          model_id=MODEL_ID)
                             (or)
            Example_URL = "https://clarifai.com/clarifai/main/models/BAAI-bge-base-en-v15"
            clarifai = ClarifaiEmbeddings(model_url=EXAMPLE_URL)
    """

    model_url: Optional[str] = None
    """Model url to use."""
    model_id: Optional[str] = None
    """Model id to use."""
    model_version_id: Optional[str] = None
    """Model version id to use."""
    app_id: Optional[str] = None
    """Clarifai application id to use."""
    user_id: Optional[str] = None
    """Clarifai user id to use."""
    pat: Optional[str] = None
    """Clarifai personal access token to use."""
    api_base: str = "https://api.clarifai.com"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that we have all required info to access Clarifai
        platform and python package exists in environment."""

        values["pat"] = get_from_dict_or_env(values, "pat", "CLARIFAI_PAT")
        user_id = values.get("user_id")
        app_id = values.get("app_id")
        model_id = values.get("model_id")
        model_url = values.get("model_url")

        if model_url is not None and model_id is not None:
            raise ValueError("Please provide either model_url or model_id, not both.")

        if model_url is None and model_id is None:
            raise ValueError("Please provide one of model_url or model_id.")

        if model_url is None and model_id is not None:
            if user_id is None or app_id is None:
                raise ValueError("Please provide a user_id and app_id.")

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Clarifai's embedding models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            from clarifai.client.input import Inputs
            from clarifai.client.model import Model
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )
        if self.pat is not None:
            pat = self.pat
        if self.model_url is not None:
            _model_init = Model(url=self.model_url, pat=pat)
        else:
            _model_init = Model(
                model_id=self.model_id,
                user_id=self.user_id,
                app_id=self.app_id,
                pat=pat,
            )

        input_obj = Inputs(pat=pat)
        batch_size = 32
        embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                input_batch = [
                    input_obj.get_text_input(input_id=str(id), raw_text=inp)
                    for id, inp in enumerate(batch)
                ]
                predict_response = _model_init.predict(input_batch)
                embeddings.extend(
                    [
                        list(output.data.embeddings[0].vector)
                        for output in predict_response.outputs
                    ]
                )

        except Exception as e:
            logger.error(f"Predict failed, exception: {e}")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Clarifai's embedding models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        try:
            from clarifai.client.model import Model
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )
        if self.pat is not None:
            pat = self.pat
        if self.model_url is not None:
            _model_init = Model(url=self.model_url, pat=pat)
        else:
            _model_init = Model(
                model_id=self.model_id,
                user_id=self.user_id,
                app_id=self.app_id,
                pat=pat,
            )

        try:
            predict_response = _model_init.predict_by_bytes(
                bytes(text, "utf-8"), input_type="text"
            )
            embeddings = [
                list(op.data.embeddings[0].vector) for op in predict_response.outputs
            ]

        except Exception as e:
            logger.error(f"Predict failed, exception: {e}")

        return embeddings[0]
