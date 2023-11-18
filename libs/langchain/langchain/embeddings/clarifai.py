import logging
from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ClarifaiEmbeddings(BaseModel, Embeddings):
    """Clarifai embedding models.

    To use, you should have the ``clarifai`` python package installed, and the
    environment variable ``CLARIFAI_PAT`` set with your personal access token or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import ClarifaiEmbeddings
            clarifai = ClarifaiEmbeddings(
                model="embed-english-light-v3.0", clarifai_api_key="my-api-key"
            )
    """

    stub: Any  #: :meta private:
    """Clarifai stub."""
    userDataObject: Any
    """Clarifai user data object."""
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
        """Validate that api key and python package exists in environment."""
        values["pat"] = get_from_dict_or_env(values, "pat", "CLARIFAI_PAT")
        user_id = values.get("user_id")
        app_id = values.get("app_id")
        model_id = values.get("model_id")

        if values["pat"] is None:
            raise ValueError("Please provide a pat.")
        if user_id is None:
            raise ValueError("Please provide a user_id.")
        if app_id is None:
            raise ValueError("Please provide a app_id.")
        if model_id is None:
            raise ValueError("Please provide a model_id.")

        try:
            from clarifai.auth.helper import ClarifaiAuthHelper
            from clarifai.client import create_stub
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )
        auth = ClarifaiAuthHelper(
            user_id=user_id,
            app_id=app_id,
            pat=values["pat"],
            base=values["api_base"],
        )
        values["userDataObject"] = auth.get_user_app_id_proto()
        values["stub"] = create_stub(auth)

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Clarifai's embedding models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        try:
            from clarifai_grpc.grpc.api import (
                resources_pb2,
                service_pb2,
            )
            from clarifai_grpc.grpc.api.status import status_code_pb2
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )

        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            post_model_outputs_request = service_pb2.PostModelOutputsRequest(
                user_app_id=self.userDataObject,
                model_id=self.model_id,
                version_id=self.model_version_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw=t))
                    )
                    for t in batch
                ],
            )
            post_model_outputs_response = self.stub.PostModelOutputs(
                post_model_outputs_request
            )

            if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
                logger.error(post_model_outputs_response.status)
                first_output_failure = (
                    post_model_outputs_response.outputs[0].status
                    if len(post_model_outputs_response.outputs)
                    else None
                )
                raise Exception(
                    f"Post model outputs failed, status: "
                    f"{post_model_outputs_response.status}, first output failure: "
                    f"{first_output_failure}"
                )
            embeddings.extend(
                [
                    list(o.data.embeddings[0].vector)
                    for o in post_model_outputs_response.outputs
                ]
            )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Clarifai's embedding models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        try:
            from clarifai_grpc.grpc.api import (
                resources_pb2,
                service_pb2,
            )
            from clarifai_grpc.grpc.api.status import status_code_pb2
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )

        post_model_outputs_request = service_pb2.PostModelOutputsRequest(
            user_app_id=self.userDataObject,
            model_id=self.model_id,
            version_id=self.model_version_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(text=resources_pb2.Text(raw=text))
                )
            ],
        )
        post_model_outputs_response = self.stub.PostModelOutputs(
            post_model_outputs_request
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            logger.error(post_model_outputs_response.status)
            first_output_failure = (
                post_model_outputs_response.outputs[0].status
                if len(post_model_outputs_response.outputs[0])
                else None
            )
            raise Exception(
                f"Post model outputs failed, status: "
                f"{post_model_outputs_response.status}, first output failure: "
                f"{first_output_failure}"
            )

        embeddings = [
            list(o.data.embeddings[0].vector)
            for o in post_model_outputs_response.outputs
        ]
        return embeddings[0]
