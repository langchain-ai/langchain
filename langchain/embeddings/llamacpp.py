"""Wrapper around llama.cpp embedding models."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.embeddings.base import Embeddings


class LlamaCppEmbeddings(BaseModel, Embeddings):
    """Wrapper around llama.cpp embedding models.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python

            from langchain.embeddings import LlamaCppEmbeddings
            llama = LlamaCppEmbeddings(model_path="/path/to/model.bin")
    """

    client: Any  #: :meta private:
    model_path: str

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into. 
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(False, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(None, alias="n_threads")
    """Number of threads to use. If None, the number 
    of threads is automatically determined."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        model_path = values["model_path"]
        n_ctx = values["n_ctx"]
        n_parts = values["n_parts"]
        seed = values["seed"]
        f16_kv = values["f16_kv"]
        logits_all = values["logits_all"]
        vocab_only = values["vocab_only"]
        use_mlock = values["use_mlock"]
        n_threads = values["n_threads"]

        try:
            from llama_cpp import Llama

            values["client"] = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_parts=n_parts,
                seed=seed,
                f16_kv=f16_kv,
                logits_all=logits_all,
                vocab_only=vocab_only,
                use_mlock=use_mlock,
                n_threads=n_threads,
                embedding=True,
            )
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )
        except Exception:
            raise NameError(f"Could not load Llama model from path: {model_path}")

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)
        return list(map(float, embedding))
