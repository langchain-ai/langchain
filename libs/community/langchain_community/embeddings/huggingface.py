from typing import Any, Dict, List, Optional, Type

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr

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

    client: Any  #: :meta private:
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

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

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

    client: Any  #: :meta private:
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

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            from InstructorEmbedding import INSTRUCTOR

            self.client = INSTRUCTOR(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )
        except ImportError as e:
            raise ImportError("Dependencies for InstructorEmbedding not found.") from e

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [[self.embed_instruction, text] for text in texts]
        embeddings = self.client.encode(instruction_pairs, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.client.encode([instruction_pair], **self.encode_kwargs)[0]
        return embedding.tolist()


class HuggingFaceBgeEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.
    To use Nomic, make sure the version of ``sentence_transformers`` >= 2.3.0.

    Bge Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en"
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

    client: Any  #: :meta private:
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

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import sentence_transformers

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence_transformers`."
            ) from exc

        self.client = sentence_transformers.SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )
        if "-zh" in self.model_name:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]
        embeddings = self.client.encode(texts, **self.encode_kwargs)
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
            self.query_instruction + text, **self.encode_kwargs
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
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.

        Example:
            .. code-block:: python

                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

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


class HuggingFaceEncoderEmbeddings(BaseModel, Embeddings):
    """HuggingFace encoder-only embedding models.
    
    To use, you must have torch and transformers be installed, 
    some examples are BERT, distil-BERT.

    Test case can be run by `python -m pytest tests/unit_tests/embeddings/test_huggingface.py`

    Example:
        .. code-block:: python

        from langchain_community.embeddings.huggingface import HuggingFaceEncoderEmbeddings

        model_name = "distilbert/distilbert-base-uncased" 
        tokenizer_name = "distilbert/distilbert-base-uncased"
        model_kwargs = {}
        tokenizer_kwargs = {"max_length": 768, "add_special_tokens": False}
        device = "cpu"
        batch_size = 2
        use_cls_embedding=False # This means only use CLS embedding as output.

        embedding = HuggingFaceEncoderEmbeddings(
            model_name=model_name, 
            tokenizer_name=tokenizer_name, 
            device=device, 
            batch_size=batch_size,
            use_cls_embedding=use_cls_embedding, 
            model_kwargs=model_kwargs, 
            tokenizer_kwargs=tokenizer_kwargs
        )
    """
    model_name: str 
    tokenizer_name: str
    device: str = "cpu"
    batch_size: int = 4
    use_cls_embedding: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = Field(default_factory=dict)

    client: Any
    tokenizer: Any
    max_length: int = 512
    add_special_tokens: bool = False
    truncation: bool = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        try:
            import torch
            from torch import Tensor
            from transformers import BatchEncoding
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError(
                "Can not import torch and transformers successfully"
            ) from e

        self.client = AutoModel\
            .from_pretrained(self.model_name, **self.model_kwargs)\
            .to(self.device)
        self.tokenizer = AutoTokenizer\
            .from_pretrained(self.tokenizer_name, **self.tokenizer_kwargs)

        if "max_length" in self.tokenizer_kwargs:
            self.max_length = self.tokenizer_kwargs["max_length"]
        if "add_special_tokens" in self.tokenizer_kwargs:
            self.add_special_tokens = self.tokenizer_kwargs["add_special_tokens"]
        if "truncation" in self.tokenizer_kwargs:
            self.truncation = self.tokenizer_kwargs["truncation"]

        torch.set_grad_enabled(False)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out_embds: List[List[float]] = []
        batch: List[str] = []
        for i, text in enumerate(texts):
            batch.append(text)
            if len(batch) == self.batch_size or i == len(texts) - 1:
                tokens: Type["BatchEncoding"] = self.tokenizer(
                    batch, 
                    padding=True, 
                    add_special_tokens=self.add_special_tokens, 
                    max_length=self.max_length, 
                    truncation=self.truncation, 
                    return_tensors='pt'
                ).to(self.device)

                self.client.eval()
                embd_vecs: Optional[Type["Tensor"]] = None
                if self.use_cls_embedding:
                    embd_vecs = self.client(**tokens)["last_hidden_state"][:, 0, :]
                else:
                    valid_length: Type["Tensor"] = tokens.attention_mask.sum(dim=1)
                    # Keep each embedding at least has valid length larger or 
                    # equal with 1
                    valid_length[valid_length == 0] = 1
                    valid_length = valid_length.reshape(valid_length.shape[0], -1)
                    
                    hidden_states: Type["Tensor"] = \
                        self.client(**tokens)["last_hidden_state"]
                    hidden_states = hidden_states * tokens.attention_mask.unsqueeze(2)
                    
                    embd_vecs = hidden_states.sum(dim=1) / valid_length

                out_embds.extend(embd_vecs.tolist())
                batch = []
        return out_embds

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

