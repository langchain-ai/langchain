from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import BaseModel, ConfigDict

LASER_MULTILINGUAL_MODEL: str = "laser2"


class LaserEmbeddings(BaseModel, Embeddings):
    """LASER Language-Agnostic SEntence Representations.
    LASER is a Python library developed by the Meta AI Research team
    and used for creating multilingual sentence embeddings for over 147 languages
    as of 2/25/2024
    See more documentation at:
    * https://github.com/facebookresearch/LASER/
    * https://github.com/facebookresearch/LASER/tree/main/laser_encoders
    * https://arxiv.org/abs/2205.12654

    To use this class, you must install the `laser_encoders` Python package.

    `pip install laser_encoders`
    Example:
        from laser_encoders import LaserEncoderPipeline
        encoder = LaserEncoderPipeline(lang="eng_Latn")
        embeddings = encoder.encode_sentences(["Hello", "World"])
    """

    lang: Optional[str] = None
    """The language or language code you'd like to use
    If empty, this implementation will default
    to using a multilingual earlier LASER encoder model (called laser2)
    Find the list of supported languages at
    https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
    """

    _encoder_pipeline: Any = None  # : :meta private:

    model_config = ConfigDict(
        extra="forbid",
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that laser_encoders has been installed."""
        try:
            from laser_encoders import LaserEncoderPipeline

            lang = values.get("lang")
            if lang:
                encoder_pipeline = LaserEncoderPipeline(lang=lang)
            else:
                encoder_pipeline = LaserEncoderPipeline(laser=LASER_MULTILINGUAL_MODEL)
            values["_encoder_pipeline"] = encoder_pipeline

        except ImportError as e:
            raise ImportError(
                "Could not import 'laser_encoders' Python package. "
                "Please install it with `pip install laser_encoders`."
            ) from e
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents using LASER.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings: np.ndarray
        embeddings = self._encoder_pipeline.encode_sentences(texts)

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate single query text embeddings using LASER.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        query_embeddings: np.ndarray
        query_embeddings = self._encoder_pipeline.encode_sentences([text])
        return query_embeddings.tolist()[0]
