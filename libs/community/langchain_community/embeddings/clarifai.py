import logging
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    pat: Optional[str] = Field(default=None, exclude=True)
    """Clarifai personal access token to use."""
    token: Optional[str] = Field(default=None, exclude=True)
    """Clarifai session token to use."""
    model: Any = Field(default=None, exclude=True)  #: :meta private:
    api_base: str = "https://api.clarifai.com"

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that we have all required info to access Clarifai
        platform and python package exists in environment."""

        try:
            from clarifai.client.model import Model
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )
        user_id = values.get("user_id")
        app_id = values.get("app_id")
        model_id = values.get("model_id")
        model_version_id = values.get("model_version_id")
        model_url = values.get("model_url")
        api_base = values.get("api_base")
        pat = values.get("pat")
        token = values.get("token")

        values["model"] = Model(
            url=model_url,
            app_id=app_id,
            user_id=user_id,
            model_version=dict(id=model_version_id),
            pat=pat,
            token=token,
            model_id=model_id,
            base_url=api_base,
        )

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Clarifai's embedding models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        from clarifai.client.input import Inputs

        input_obj = Inputs.from_auth_helper(self.model.auth_helper)
        batch_size = 32
        embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                input_batch = [
                    input_obj.get_text_input(input_id=str(id), raw_text=inp)
                    for id, inp in enumerate(batch)
                ]
                predict_response = self.model.predict(input_batch)
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
            predict_response = self.model.predict_by_bytes(
                bytes(text, "utf-8"), input_type="text"
            )
            embeddings = [
                list(op.data.embeddings[0].vector) for op in predict_response.outputs
            ]

        except Exception as e:
            logger.error(f"Predict failed, exception: {e}")

        return embeddings[0]
