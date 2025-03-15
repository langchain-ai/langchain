"""Ollama embeddings models."""

from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from ollama import AsyncClient, Client
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self


class OllamaEmbeddings(BaseModel, Embeddings):
    """Ollama embedding model integration.

    Set up a local Ollama instance:
        Install the Ollama package and set up a local Ollama instance
        using the instructions here: https://github.com/ollama/ollama .

        You will need to choose a model to serve.

        You can view a list of available models via the model library (https://ollama.com/library).

        To fetch a model from the Ollama model library use ``ollama pull <name-of-model>``.

        For example, to pull the llama3 model:

        .. code-block:: bash

            ollama pull llama3

        This will download the default tagged version of the model.
        Typically, the default points to the latest, smallest sized-parameter model.

        * On Mac, the models will be downloaded to ~/.ollama/models
        * On Linux (or WSL), the models will be stored at /usr/share/ollama/.ollama/models

        You can specify the exact version of the model of interest
        as such ``ollama pull vicuna:13b-v1.5-16k-q4_0``.

        To view pulled models:

        .. code-block:: bash

            ollama list

        To start serving:

        .. code-block:: bash

            ollama serve

        View the Ollama documentation for more commands.

        .. code-block:: bash

            ollama help

    Install the langchain-ollama integration package:
        .. code-block:: bash

            pip install -U langchain_ollama

    Key init args — completion params:
        model: str
            Name of Ollama model to use.
        base_url: Optional[str]
            Base url the model is hosted under.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_ollama import OllamaEmbeddings

            embed = OllamaEmbeddings(
                model="llama3"
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Embed multiple texts:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Async:
        .. code-block:: python

            vector = await embed.aembed_query(input_text)
           print(vector[:3])

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
    """  # noqa: E501

    model: str
    """Model name to use."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx Client. 
    For a full list of the params, see [this link](https://pydoc.dev/httpx/latest/httpx.Client.html)
    """

    _client: Client = PrivateAttr(default=None)  # type: ignore
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = PrivateAttr(default=None)  # type: ignore
    """
    The async client to use for making requests.
    """

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    keep_alive: Optional[int] = None
    """controls how long the model will stay loaded into memory
    following the request (default: 5m)
    """

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    model_config = ConfigDict(
        extra="forbid",
    )

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "stop": self.stop,
            "tfs_z": self.tfs_z,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients to use for ollama."""
        client_kwargs = self.client_kwargs or {}
        self._client = Client(host=self.base_url, **client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **client_kwargs)
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = self._client.embed(
            self.model, texts, options=self._default_params, keep_alive=self.keep_alive
        )["embeddings"]
        return embedded_docs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = (
            await self._async_client.embed(
                self.model, texts, keep_alive=self.keep_alive
            )
        )["embeddings"]
        return embedded_docs

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
