import importlib
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict
from langchain_core.utils import pre_init


class EmbedAnythingEmbeddings(BaseModel, Embeddings):
    """EmbedAnything Embeddings model.
    EmbedAnything is a Python library for embedding generation.
    See more documentation at:
    
    To use this class, you must install the `embed_anything` Python package.
    
    `pip install embed_anything`

    For GPU support, you can install the 

    `embed_anything-gpu` package.
    Example:
    from langchain_community.embeddings import EmbedAnythingEmbeddings
    embed_anything = EmbedAnythingEmbeddings()
    """

    model: str = "jina"
    """The model to use for embedding generation. Defaults to "jina".
    The available options are: "bert", "jina", "colbert", "sparse-bert
    """

    model_id: str = "jinaai/jina-embeddings-v2-base-en"
    """The model id to use for embedding generation. Defaults to "jinaai/jina-embeddings-v2-base-en".
    The available options are:
    * "bert": "bert-base-uncased"
    * "jina": "jinaai/jina-embeddings-v2-base-en"
    * "colbert": "castorini/monobert-large-msmarco"
    * "sparse-bert": "castorini/sparsebert-uncased-msmarco"
    """

    batch_size: int = 256
    """Batch size for encoding. Higher values will use more memory, but be faster. Defaults to 256.
    """

    backend: str = "candle"
    """The backend to use for embedding generation. Defaults to "candle".
    The available options are: "candle", "onnx"
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    """The model configuration. Defaults to None.
    """

    embedder: Any = None
    """The embedder object. Defaults to None.
    """

    embed_anything: Any = None
    """The embed_anything module. Defaults to None."""

    config: Any = None
    """The TextEmbedConfig object. Defaults to None."""

    path_in_repo:str = "model.onnx"
    """The path to the onnx model in the repository. Defaults to "model.onnx"."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:

        try:
            import embed_anything
            from embed_anything import EmbeddingModel, TextEmbedConfig, WhichModel

            values["embed_anything"] = importlib.import_module("embed_anything")
        except ImportError:
            raise ImportError(
                "You need to install the 'embed_anything' package to use this embedding model. For gpu support, you can install the 'embed_anything-gpu' package."
            )

        model = values.get("model")
        model_id = values.get("model_id")
        backend = values.get("backend")
        path_in_repo = values.get("path_in_repo")

        if model == "bert":
            model_type = WhichModel.Bert
        elif model == "jina":
            model_type = WhichModel.Jina
        elif model == "colbert":
            model_type = WhichModel.ColBert
        elif model == "sparse-bert":
            model_type = WhichModel.SparseBert

        if backend == "candle":
            model = EmbeddingModel.from_pretrained_hf(
                model=model_type, model_id=model_id
            )

        elif backend == "onnx":
            print(path_in_repo)

            model = EmbeddingModel.from_pretrained_onnx(
                model=model_type, hf_model_id=model_id, path_in_repo=path_in_repo
            )

        values["config"] = TextEmbedConfig(batch_size=values.get("batch_size"))

        values["embedder"] = model
        return values

    def embed_documents(self, texts: list[str]):

        embed_data = self.embed_anything.embed_query(
            texts, self.embedder, config=self.config
        )
        return [e.embedding for e in embed_data]

    def embed_query(self, text: str):

        embed_data = self.embed_anything.embed_query(
            [text], self.embedder, config=self.config
        )[0]
        return embed_data.embedding
