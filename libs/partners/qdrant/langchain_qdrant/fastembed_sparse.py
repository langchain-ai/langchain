from typing import Any, List, Optional, Sequence

from pydantic import Field

from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector


class FastEmbedSparse(SparseEmbeddings):
    """An interface for sparse embedding models to use with Qdrant."""

    def __init__(
        self,
        model_name: str = Field(default="Qdrant/bm25", alias="model"),
        batch_size: int = 256,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[Any]] = None,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Sparse encoder implementation using FastEmbed - https://qdrant.github.io/fastembed/
        For a list of available models, see https://qdrant.github.io/fastembed/examples/Supported_Models/

        Args:
            model_name (str): The name of the model to use. Defaults to `"Qdrant/bm25"`.
            batch_size (int): Batch size for encoding. Defaults to 256.
            cache_dir (str, optional): The path to the model cache directory.\
                                       Can also be set using the\
                                       `FASTEMBED_CACHE_PATH` env variable.
            threads (int, optional): The number of threads onnxruntime session can use.
            providers (Sequence[Any], optional): List of ONNX execution providers.\
            parallel (int, optional): If `>1`, data-parallel encoding will be used, r\
                                      Recommended for encoding of large datasets.\
                                      If `0`, use all available cores.\
                                      If `None`, don't use data-parallel processing,\
                                      use default onnxruntime threading instead.\
                                      Defaults to None.
            kwargs: Additional options to pass to fastembed.SparseTextEmbedding
        Raises:
            ValueError: If the model_name is not supported in SparseTextEmbedding.
        """
        try:
            from fastembed import SparseTextEmbedding  # type: ignore
        except ImportError:
            raise ValueError(
                "The 'fastembed' package is not installed. "
                "Please install it with "
                "`pip install fastembed` or `pip install fastembed-gpu`."
            )
        self._batch_size = batch_size
        self._parallel = parallel
        self._model = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        results = self._model.embed(
            texts, batch_size=self._batch_size, parallel=self._parallel
        )
        return [
            SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
            for result in results
        ]

    def embed_query(self, text: str) -> SparseVector:
        result = next(self._model.query_embed(text))

        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )
