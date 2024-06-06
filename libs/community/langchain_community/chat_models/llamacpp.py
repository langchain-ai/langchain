from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Iterator,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
    Mapping
)
if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from pathlib import Path
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)

from langchain_openai.chat_models.base import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict
)


from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, root_validator, BaseModel
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.utils.function_calling import convert_to_openai_tool


class ChatLlamaCpp(BaseChatModel):
    """llama.cpp model.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    """

    client: Any  #: :meta private:

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

    rope_freq_scale: float = 1.0
    """Scale factor for rope sampling."""

    rope_freq_base: float = 10000.0
    """Base frequency for rope sampling."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Any additional parameters to pass to llama_cpp.Llama."""

    streaming: bool = True
    """Whether to stream the results, token by token."""

    grammar_path: Optional[Union[str, Path]] = None
    """
    grammar_path: Path to the .gbnf file that defines formal grammars
    for constraining model outputs. For instance, the grammar can be used
    to force the model to generate valid JSON or to speak exclusively in emojis. At most
    one of grammar_path and grammar should be passed in.
    """
    grammar: Optional[Union[str, Any]] = None
    """
    grammar: formal grammar for constraining model outputs. For instance, the grammar 
    can be used to force the model to generate valid JSON or to speak exclusively in 
    emojis. At most one of grammar_path and grammar should be passed in.
    """

    verbose: bool = True
    """Print verbose output to stderr."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError:
            raise ImportError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )

        model_path = values["model_path"]
        model_param_names = [
            "rope_freq_scale",
            "rope_freq_base",
            "lora_path",
            "lora_base",
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "use_mmap",
            "last_n_tokens_size",
            "verbose",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]

        model_params.update(values["model_kwargs"])

        try:
            values["client"] = Llama(model_path, **model_params)
        except Exception as e:
            raise ValueError(
                f"Could not load Llama model from path: {model_path}. "
                f"Received error {e}"
            )

        if values["grammar"] and values["grammar_path"]:
            grammar = values["grammar"]
            grammar_path = values["grammar_path"]
            raise ValueError(
                "Can only pass in one of grammar and grammar_path. Received "
                f"{grammar=} and {grammar_path=}."
            )
        elif isinstance(values["grammar"], str):
            values["grammar"] = LlamaGrammar.from_string(values["grammar"])
        elif values["grammar_path"]:
            values["grammar"] = LlamaGrammar.from_file(values["grammar_path"])
        else:
            pass
        return values
    

    def _get_parameters(self) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters in format needed by llama_cpp.

        Returns:
            Dictionary containing the combined parameters.
        """

        params = self._default_params

        # llama_cpp expects the "stop" key not this, so we remove it:
        params.pop("stop_sequences")

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stop or []

        return params


    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        return message_dicts


    def _create_chat_result(
        self, response: dict
    ) -> ChatResult:
        generations = []

        for res in response["choices"]:
            message = _convert_dict_to_message(res["text"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)


    def _generate(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        if self.streaming:
            stream_iter = self._stream(
                messages, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        
        params = {**self._get_parameters(), **kwargs}
        message_dicts = self._create_message_dicts(messages)

        response = self.client.create_chat_completion(messages=message_dicts, **params)

        return self._create_chat_result(response)
    

    def _stream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        params = {**self._get_parameters(), **kwargs}
        message_dicts = self._create_message_dicts(messages)

        result = self.client.create_chat_completion(messages=message_dicts, stream=True, **params)
        
        default_chunk_class = AIMessageChunk
        count = 0
        for chunk in result:
            count += 1
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text, chunk=chunk, logprobs=logprobs
                )
            yield chunk

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: Optional[Dict[str, str]]= None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model
        
        tool_choice: temporarily not support for "any", "auto" choices like OpenAI tool-calling API.
                    should be a dict of the form to force this tool
                    {"type": "function", "function": {"name": <<tool_name>>}}.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    """Unrecognized tool_choice type. Expected dict having format like this {"type": "function", "function": {"name": <<tool_name>>}}"""
                    f"Received: {tool_choice}"
                )

        kwargs["tool_choice"] = tool_choice
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)



    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            **{"model_path": self.model_path}, **self._default_params
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llama-cpp-python"
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling create_chat_completion."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "logprobs": self.logprobs,
            "stop_sequences": self.stop,  # key here is convention among LLM classes
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
        }
        if self.grammar:
            params["grammar"] = self.grammar
        return params


