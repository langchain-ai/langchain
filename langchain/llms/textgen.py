"""Wrapper around llama.cpp."""
import logging
import requests
from typing import Any, Dict, Generator, List, Optional

from pydantic import Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)


class TextGen(LLM):
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

    model_url: str
    max_new_tokens: Optional[int] = 250
    do_sample: bool = Field(True, alias="do_sample")
    temperature: Optional[float] = 1.3
    top_p: Optional[float] = 0.1
    typical_p: Optional[float] = 1
    epsilon_cutoff: Optional[float] = 0  # In units of 1e-4
    eta_cutoff: Optional[float] = 0  # In units of 1e-4
    repetition_penalty: Optional[float] = 1.18
    top_k: Optional[float] = 40
    min_length: Optional[int] = 0
    no_repeat_ngram_size: Optional[int] = 0
    num_beams: Optional[int] = 1
    penalty_alpha: Optional[float] = 0
    length_penalty: Optional[float] = 1
    early_stopping: bool = Field(False, alias="early_stopping")
    seed: int = Field(-1, alias="seed")
    add_bos_token: bool = Field(True, alias="add_bos_token")
    truncation_length: Optional[int] = 2048
    ban_eos_token: bool = Field(False, alias="ban_eos_token")
    skip_special_tokens: bool = Field(True, alias="skip_special_tokens")
    stopping_strings: Optional[List[str]] = []
    streaming: bool = False

    '''
    model_path: str
    """The path to the Llama model file."""

    lora_base: Optional[str] = None
    """The path to the Llama LoRA base model."""

    lora_path: Optional[str] = None
    """The path to the Llama LoRA. If None, no LoRa is loaded."""

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into.
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(True, alias="f16_kv")
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

    n_gpu_layers: Optional[int] = Field(None, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

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

    use_mmap: Optional[bool] = True
    """Whether to keep the model loaded in RAM"""

    streaming: bool = True
    """Whether to stream the results, token by token."""
    '''

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        model_url = values["model_url"]
        # model_path = values["model_path"]
        model_param_names = [
            'max_new_tokens',
            'do_sample',
            'temperature',
            'top_p',
            'typical_p',
            'epsilon_cutoff',
            'eta_cutoff',
            'repetition_penalty',
            'top_k',
            'min_length',
            'no_repeat_ngram_size',
            'num_beams',
            'penalty_alpha',
            'length_penalty',
            'early_stopping',
            'seed',
            'add_bos_token',
            'truncation_length',
            'ban_eos_token',
            'skip_special_tokens',
            'stopping_strings',
        ]
        model_params = {k: values[k] for k in model_param_names}

        values["client"] = True
        # try:
        #     from llama_cpp import Llama

        #     values["client"] = Llama(model_path, **model_params)
        # except Exception as e:
        #     raise ValueError(
        #         # f"Could not load Llama model from path: {model_path}. "
        #         f"Received error {e}"
        #     )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling llama_cpp."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "stopping_strings": self.stop,  # key here is convention among LLM classes
            "stopping_strings": self.stopping_strings,  # key here is convention among LLM classes
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_url": self.model_url}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "textgen"

    def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Performs sanity check, preparing paramaters in format needed by llama_cpp.

        Args:
            stop (Optional[List[str]]): List of stop sequences for llama_cpp.

        Returns:
            Dictionary containing the combined parameters.
        """

        # Raise error if stop sequences are in both input and default params
        # if self.stop and stop is not None:
        if self.stopping_strings and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")

        params = self._default_params

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stopping_strings or stop or []

        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call the Llama model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain.llms import LlamaCpp
                llm = LlamaCpp(model_path="/path/to/local/llama/model.bin")
                llm("This is a prompt.")
        """
        if self.streaming:
            # If streaming is enabled, we use the stream
            # method that yields as they are generated
            # and return the combined strings from the first choices's text:
            combined_text_output = ""
            for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
                combined_text_output += token["choices"][0]["text"]
            return combined_text_output
        else:
            params = self._get_parameters(stop)
            # result = self.client(prompt=prompt, **params)
            request = params.copy()
            # del(request["model_url"])
            request["prompt"] = prompt
            response = requests.post(self.model_url, json=request)

            if response.status_code == 200:
                result = response.json()['results'][0]['text']
                print(prompt + result)
            else:
                print(f"ERROR: response: {response}")

            return result

    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:
        """Yields results objects as they are generated in real time.

        BETA: this is a beta feature while we figure out the right abstraction:
        Once that happens, this interface could change.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See llama-cpp-python docs and below for more.

        Example:
            .. code-block:: python

                from langchain.llms import LlamaCpp
                llm = LlamaCpp(
                    model_path="/path/to/local/model.bin",
                    temperature = 0.5
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                        stop=["'","\n"]):
                    result = chunk["choices"][0]
                    print(result["text"], end='', flush=True)

        """
        params = self._get_parameters(stop)
        result = self.client(prompt=prompt, stream=True, **params)
        for chunk in result:
            token = chunk["choices"][0]["text"]
            log_probs = chunk["choices"][0].get("logprobs", None)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=token, verbose=self.verbose, log_probs=log_probs
                )
            yield chunk
