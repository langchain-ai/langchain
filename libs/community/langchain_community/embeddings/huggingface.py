import warnings
from typing import Any, Dict, List, Optional

import requests
from langchain_core._api import deprecated, warn_deprecated
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_BGE_MODEL = "BAAI/bge-large-en"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


@deprecated(
    since="0.2.2",
    removal="1.0",
    alternative_import="langchain_huggingface.HuggingFaceEmbeddings",
)
class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any = None  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the Sentence
    Transformer model, such as `prompt_name`, `prompt`, `batch_size`, `precision`,
    `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        if "model_name" not in kwargs:
            since = "0.2.16"
            removal = "0.4.0"
            warn_deprecated(
                since=since,
                removal=removal,
                message=f"Default values for {self.__class__.__name__}.model_name"
                + f" were deprecated in LangChain {since} and will be removed in"
                + f" {removal}. Explicitly pass a model_name to the"
                + f" {self.__class__.__name__} constructor instead.",
            )

        try:
            import sentence_transformers

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self.client = sentence_transformers.SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(
                texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


class HuggingFaceInstructEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers``
    and ``InstructorEmbedding`` python packages installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceInstructEmbeddings

            model_name = "hkunlp/instructor-large"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceInstructEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any = None  #: :meta private:
    model_name: str = DEFAULT_INSTRUCT_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding documents."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        if "model_name" not in kwargs:
            since = "0.2.16"
            removal = "0.4.0"
            warn_deprecated(
                since=since,
                removal=removal,
                message=f"Default values for {self.__class__.__name__}.model_name"
                + f" were deprecated in LangChain {since} and will be removed in"
                + f" {removal}. Explicitly pass a model_name to the"
                + f" {self.__class__.__name__} constructor instead.",
            )

        try:
            from InstructorEmbedding import INSTRUCTOR

            self.client = INSTRUCTOR(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )
        except ImportError as e:
            raise ImportError("Dependencies for InstructorEmbedding not found.") from e

        if "show_progress_bar" in self.encode_kwargs:
            warn_deprecated(
                since="0.2.5",
                removal="1.0",
                name="encode_kwargs['show_progress_bar']",
                alternative=f"the show_progress method on {self.__class__.__name__}",
            )
            if self.show_progress:
                warnings.warn(
                    "Both encode_kwargs['show_progress_bar'] and show_progress are set;"
                    "encode_kwargs['show_progress_bar'] takes precedence"
                )
            self.show_progress = self.encode_kwargs.pop("show_progress_bar")

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [[self.embed_instruction, text] for text in texts]
        embeddings = self.client.encode(
            instruction_pairs,
            show_progress_bar=self.show_progress,
            **self.encode_kwargs,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.client.encode(
            [instruction_pair],
            show_progress_bar=self.show_progress,
            **self.encode_kwargs,
        )[0]
        return embedding.tolist()


@deprecated(
    since="0.2.2",
    removal="1.0",
    alternative_import="langchain_huggingface.HuggingFaceEmbeddings",
)
class HuggingFaceBgeEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.
    To use Nomic, make sure the version of ``sentence_transformers`` >= 2.3.0.

    Bge Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
     Nomic Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            model_name = "nomic-ai/nomic-embed-text-v1"
            model_kwargs = {
                'device': 'cpu',
                'trust_remote_code':True
                }
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction = "search_query:",
                embed_instruction = "search_document:"
            )
    """

    client: Any = None  #: :meta private:
    model_name: str = DEFAULT_BGE_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""
    embed_instruction: str = ""
    """Instruction to use for embedding document."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        if "model_name" not in kwargs:
            since = "0.2.5"
            removal = "0.4.0"
            warn_deprecated(
                since=since,
                removal=removal,
                message=f"Default values for {self.__class__.__name__}.model_name"
                + f" were deprecated in LangChain {since} and will be removed in"
                + f" {removal}. Explicitly pass a model_name to the"
                + f" {self.__class__.__name__} constructor instead.",
            )

        try:
            import sentence_transformers

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc
        extra_model_kwargs = [
            "torch_dtype",
            "attn_implementation",
            "provider",
            "file_name",
            "export",
        ]
        extra_model_kwargs_dict = {
            k: self.model_kwargs.pop(k)
            for k in extra_model_kwargs
            if k in self.model_kwargs
        }
        self.client = sentence_transformers.SentenceTransformer(
            self.model_name,
            cache_folder=self.cache_folder,
            **self.model_kwargs,
            model_kwargs=extra_model_kwargs_dict,
        )

        if "-zh" in self.model_name:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

        if "show_progress_bar" in self.encode_kwargs:
            warn_deprecated(
                since="0.2.5",
                removal="1.0",
                name="encode_kwargs['show_progress_bar']",
                alternative=f"the show_progress method on {self.__class__.__name__}",
            )
            if self.show_progress:
                warnings.warn(
                    "Both encode_kwargs['show_progress_bar'] and show_progress are set;"
                    "encode_kwargs['show_progress_bar'] takes precedence"
                )
            self.show_progress = self.encode_kwargs.pop("show_progress_bar")

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]
        embeddings = self.client.encode(
            texts, show_progress_bar=self.show_progress, **self.encode_kwargs
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client.encode(
            self.query_instruction + text,
            show_progress_bar=self.show_progress,
            **self.encode_kwargs,
        )
        return embedding.tolist()


class HuggingFaceInferenceAPIEmbeddings(BaseModel, Embeddings):
    """Embed texts using the HuggingFace API.

    Requires a HuggingFace Inference API key and a model name.
    """

    api_key: SecretStr
    """Your API key for the HuggingFace Inference API."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """The name of the model to use for text embeddings."""
    api_url: Optional[str] = None
    """Custom inference endpoint url. None for using default public url."""
    additional_headers: Dict[str, str] = {}
    """Pass additional headers to the requests library if needed."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @property
    def _api_url(self) -> str:
        return self.api_url or self._default_api_url

    @property
    def _default_api_url(self) -> str:
        return (
            "https://api-inference.huggingface.co"
            "/pipeline"
            "/feature-extraction"
            f"/{self.model_name}"
        )

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            **self.additional_headers,
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.

        Example:
            .. code-block:: python

                from langchain_community.embeddings import (
                    HuggingFaceInferenceAPIEmbeddings,
                )

                hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key="your_api_key",
                    model_name="sentence-transformers/all-MiniLM-l6-v2"
                )
                texts = ["Hello, world!", "How are you?"]
                hf_embeddings.embed_documents(texts)
        """  # noqa: E501
        response = requests.post(
            self._api_url,
            headers=self._headers,
            json={
                "inputs": texts,
                "options": {"wait_for_model": True, "use_cache": True},
            },
        )
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
