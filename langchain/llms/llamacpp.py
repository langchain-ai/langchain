"""Wrapper around llama.cpp."""
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Coroutine, Generator, Iterable, AsyncIterable, AsyncGenerator
import queue

from pydantic import BaseModel, Field, root_validator

from langchain.llms.base import LLM, Generation, LLMResult

logger = logging.getLogger(__name__)


class LlamaCpp(LLM):
    """Wrapper around the llama.cpp model.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python

            from langchain.llms import LlamaCppEmbeddings
            llm = LlamaCppEmbeddings(model_path="/path/to/llama/model")
    """

    client: Any  #: :meta private:
    model_path: str
    """The path to the Llama model file."""

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
    """Number of threads to use. 
    If None, the number of threads is automatically determined."""

    n_batch: Optional[int] = Field(8, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    suffix: Optional[str] = Field(None)
    """A suffix to append to the generated text. If None, no suffix is appended."""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0.8
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling."""

    logprobs: Optional[int] = Field(None)
    """The number of logprobs to return. If None, no logprobs are returned."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    repeat_penalty: Optional[float] = 1.1
    """The penalty to apply to repeated tokens."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    last_n_tokens_size: Optional[int] = 64
    """The number of tokens to look back when applying the repeat_penalty."""

    verbose: Optional[bool] = Field(default=False)

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
        n_batch = values["n_batch"]
        last_n_tokens_size = values["last_n_tokens_size"]

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
                n_batch=n_batch,
                last_n_tokens_size=last_n_tokens_size,
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

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling llama_cpp."""
        return {
            "suffix": self.suffix,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "echo": self.echo,
            "stop_sequences": self.stop,
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
        }

    def _prepare_params(self, stop: Optional[List[str]] = None, **config: Any) -> Dict[str, Any]:
        """Prepare the parameters for calling llama_cpp, using defaults and input while checking for conflicts."""
        params = self._default_params
        if self.stop and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop:
            params["stop_sequences"] = self.stop
        else:
            params["stop_sequences"] = []
        
        input_args = {
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "logprobs": params["logprobs"],
            "echo": params["echo"],
            "stop": params["stop_sequences"],
            "repeat_penalty": params["repeat_penalty"],
            "top_k": params["top_k"],
        }
        input_args.update(config)
        return input_args

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_path": self.model_path}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama.cpp"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **config: Any) -> str:
        """Call the Llama model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain.llms import LlamaCppEmbeddings
                llm = LlamaCppEmbeddings(model_path="/path/to/local/llama/model.bin")
                llm("This is a prompt.")
        """

        params = self._prepare_params(stop, **config)
        text = self.client(
            prompt=prompt,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            logprobs=params["logprobs"],
            echo=params["echo"],
            stop=params["stop_sequences"],
            repeat_penalty=params["repeat_penalty"],
            top_k=params["top_k"],
        )
        return text["choices"][0]["text"]

    async def _agenerate(
        self, prompts: List[str], 
        stop: Optional[List[str]] = None, 
        stream: bool = True, 
        **config: Any
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous generator for streaming tokens from the Llama models.

        Args:
            prompts: A list of prompts to use for generation.
            stop: A list of strings to stop generation when encountered.
            stream: Whether to stream tokens as they are generated.
            config: Additional configuration parameters for the client.

        Yields:
            An async iterable of generated token strings.
        """
        params = self._prepare_params(stop, **config)

        if stream:
            for generation in self.client(prompt=prompts[0], stream=stream, **params):
                chunk = generation['choices'][0]['text']
            if self.verbose:
                print(chunk, end="", flush=True)
            yield chunk
        else:
            result = self.client(prompt=prompts[0], stream=stream, **params)
            text = result['choices'][0]['text']
            if self.verbose:
                print(text)
            yield text

    def _run_coroutine(self, coroutine: Coroutine) -> Generator[str, None, None]:
        """
        Executes the given coroutine and yields individual tokens.

        Args:
            coroutine: The coroutine to execute.

        Yields:
            Individual tokens as strings.
        """
        chunk_queue = queue.Queue()

        async def _enqueue_chunks():
            async for chunk in coroutine:
                chunk_queue.put(chunk)
            chunk_queue.put(None)

        def _run_coroutine_internal():
            asyncio.run(_enqueue_chunks())

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_coroutine_internal)

        while True:
            chunk = chunk_queue.get(timeout=60)
            if chunk is None:
                break
            yield chunk

        future.result()
        executor.shutdown(wait=True)
    def stream(self, prompt: str, stop: Optional[List[str]] = None, **config: Any) -> Iterable[str]:
        """
        Stream tokens from the Llama model.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.
            config: Additional configuration parameters for the client.

        Returns:
            An iterable of generated tokens.
        """
        prepared_params = self._prepare_params(stop, **config)
        generator = self._agenerate([prompt], **prepared_params)
        return self._run_coroutine(generator)

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        generator = self._agenerate(prompts, stop=stop)
        result = LLMResult(generations=[[Generation(text="")]])
        
        for token in self._run_coroutine(generator):
            result.generations[0][0].text += token
            self.callback_manager.on_llm_new_token(
                {"choices": [{"text": token}]},
                verbose=self.verbose,
            )

        return result

