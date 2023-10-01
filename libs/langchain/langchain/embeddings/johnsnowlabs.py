import os
import sys
from typing import Any, List

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra


class JohnSnowLabsEmbeddings(BaseModel, Embeddings):
    """JohnSnowLabs embedding models

    To use, you should have the ``johnsnowlabs`` python package installed.
    Example:        .. code-block:: python
            from langchain.embeddings.johnsnowlabs import JohnSnowLabsEmbeddings
            document = "foo bar"
            embedding = JohnSnowLabsEmbeddings('embed_sentence.bert')
            output = embedding.embed_query(document)
    """

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("JohnSnowLabsEmbeddings does not support async yet")

    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError("JohnSnowLabsEmbeddings does not support async yet")

    model: Any

    def __init__(self, model='embed_sentence.bert', hardware_target='cpu', **kwargs: Any):
        """Initialize the johnsnowlabs model."""
        super().__init__(**kwargs)
        # 1) Check imports
        try:
            from johnsnowlabs import nlp
            from nlu.pipe.pipeline import NLUPipeline
        except ImportError as exc:
            raise ImportError(
                "Could not import johnsnowlabs python package. "
                "Please install it with `pip install johnsnowlabs`.") from exc

            # 2) Start a Spark Session
        try:
            os.environ['PYSPARK_PYTHON'] = sys.executable
            os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
            nlp.start(hardware_target=hardware_target)
        except Exception as exc:
            raise Exception('Failure starting Spark Session') from exc
            # 3) Load the model
        try:
            if isinstance(model, str):
                self.model = nlp.load(model)
            elif isinstance(model, NLUPipeline):
                self.model = model
            else:
                self.model = nlp.to_nlu_pipe(model)
        except Exception as exc:
            raise Exception('Failure loading model') from exc

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a JohnSnowLabs transformer model.

        Args:            texts: The list of texts to embed.
        Returns:            List of embeddings, one for each text.        """

        return _select_embed_document(self.model.predict(texts, output_level='document'))

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a JohnSnowLabs transformer model.
        Args:            text: The text to embed.
        Returns:            Embeddings for the text.        """
        return _select_embed_query(self.model.predict(text, output_level='document'))


def _select_embed_document(df):
    emb_col = None
    for c in df.columns:
        if 'embedding' in c:
            emb_col = c
    return [vec.tolist() for vec in df[emb_col].tolist()]


def _select_embed_query(df):
    emb_col = None
    for c in df.columns:
        if 'embedding' in c:
            emb_col = c
    return [vec.tolist() for vec in df[emb_col].tolist()][0]
