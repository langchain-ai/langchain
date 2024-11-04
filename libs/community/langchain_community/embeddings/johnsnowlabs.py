import os
import sys
from typing import Any, List

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict


class JohnSnowLabsEmbeddings(BaseModel, Embeddings):
    """JohnSnowLabs embedding models

    To use, you should have the ``johnsnowlabs`` python package installed.
    Example:
        .. code-block:: python

            from langchain_community.embeddings.johnsnowlabs import JohnSnowLabsEmbeddings

            embedding = JohnSnowLabsEmbeddings(model='embed_sentence.bert')
            output = embedding.embed_query("foo bar")
    """  # noqa: E501

    model: Any = "embed_sentence.bert"

    def __init__(
        self,
        model: Any = "embed_sentence.bert",
        hardware_target: str = "cpu",
        **kwargs: Any,
    ):
        """Initialize the johnsnowlabs model."""
        super().__init__(**kwargs)
        # 1) Check imports
        try:
            from johnsnowlabs import nlp
            from nlu.pipe.pipeline import NLUPipeline
        except ImportError as exc:
            raise ImportError(
                "Could not import johnsnowlabs python package. "
                "Please install it with `pip install johnsnowlabs`."
            ) from exc

        # 2) Start a Spark Session
        try:
            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
            nlp.start(hardware_target=hardware_target)
        except Exception as exc:
            raise Exception("Failure starting Spark Session") from exc

        # 3) Load the model
        try:
            if isinstance(model, str):
                self.model = nlp.load(model)
            elif isinstance(model, NLUPipeline):
                self.model = model
            else:
                self.model = nlp.to_nlu_pipe(model)
        except Exception as exc:
            raise Exception("Failure loading model") from exc

    model_config = ConfigDict(
        extra="forbid",
    )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a JohnSnowLabs transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        df = self.model.predict(texts, output_level="document")
        emb_col = None
        for c in df.columns:
            if "embedding" in c:
                emb_col = c
        return [vec.tolist() for vec in df[emb_col].tolist()]

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a JohnSnowLabs transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
