from __future__ import annotations

import importlib.util
import platform
from collections.abc import AsyncIterator
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    get_origin,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal

from langchain_community.adapters.openai import convert_message_to_dict

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]


class ChatOutlines(BaseChatModel):
    """Outlines chat model integration.

    Setup:
      pip install outlines

    Key init args — client params:
      backend: Literal["llamacpp", "transformers", "transformers_vision", "vllm", "mlxlm"] = "transformers"
        Specifies the backend to use for the model.

    Key init args — completion params:
      model: str
        Identifier for the model to use with Outlines.
      max_tokens: int = 256
        The maximum number of tokens to generate.
      stop: Optional[List[str]] = None
        A list of strings to stop generation when encountered.
      streaming: bool = True
        Whether to stream the results, token by token.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
      from langchain_community.chat_models import ChatOutlines
      chat = ChatOutlines(model="meta-llama/Llama-2-7b-chat-hf")

    Invoke:
      chat.invoke([HumanMessage(content="Say foo:")])

    Stream:
      for chunk in chat.stream([HumanMessage(content="Count to 10:")]):
          print(chunk.content, end="", flush=True)

    """  # noqa: E501

    client: Any = None  # :meta private:

    model: str
    """Identifier for the model to use with Outlines.
    
    The model identifier should be a string specifying:
    - A Hugging Face model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
    - A local path to a model
    - For GGUF models, the format is "repo_id/file_name"
      (e.g., "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf")
    
    Examples:
    - "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    - "meta-llama/Llama-2-7b-chat-hf"
    """

    backend: Literal[
        "llamacpp", "transformers", "transformers_vision", "vllm", "mlxlm"
    ] = "transformers"
    """Specifies the backend to use for the model.
    
    Supported backends are:
    - "llamacpp": For GGUF models using llama.cpp
    - "transformers": For Hugging Face Transformers models (default)
    - "transformers_vision": For vision-language models (e.g., LLaVA)
    - "vllm": For models using the vLLM library
    - "mlxlm": For models using the MLX framework
    
    Note: Ensure you have the necessary dependencies installed for the chosen backend.
    The system will attempt to import required packages and may raise an ImportError
    if they are not available.
    """

    max_tokens: int = 256
    """The maximum number of tokens to generate."""

    stop: Optional[List[str]] = None
    """A list of strings to stop generation when encountered."""

    streaming: bool = True
    """Whether to stream the results, token by token."""

    regex: Optional[str] = None
    """Regular expression for structured generation.
    
    If provided, Outlines will guarantee that the generated text matches this regex.
    This can be useful for generating structured outputs like IP addresses, dates, etc.
    
    Example: (valid IP address)
        regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    
    Note: Computing the regex index can take some time, so it's recommended to reuse
    the same regex for multiple generations if possible.
    
    For more details, see: https://dottxt-ai.github.io/outlines/reference/generation/regex/
    """

    type_constraints: Optional[Union[type, str]] = None
    """Type constraints for structured generation.
    
    Restricts the output to valid Python types. Supported types include:
    int, float, bool, datetime.date, datetime.time, datetime.datetime.
    
    Example:
        type_constraints = int
    
    For more details, see: https://dottxt-ai.github.io/outlines/reference/generation/format/
    """

    json_schema: Optional[Union[Any, Dict, Callable]] = None
    """Pydantic model, JSON Schema, or callable (function signature)
    for structured JSON generation.
    
    Outlines can generate JSON output that follows a specified structure,
    which is useful for:
    1. Parsing the answer (e.g., with Pydantic), storing it, or returning it to a user.
    2. Calling a function with the result.

    You can provide:
    - A Pydantic model
    - A JSON Schema (as a Dict)
    - A callable (function signature)

    The generated JSON will adhere to the specified structure.

    For more details, see: https://dottxt-ai.github.io/outlines/reference/generation/json/
    """

    grammar: Optional[str] = None
    """Context-free grammar for structured generation.
    
    If provided, Outlines will generate text that adheres to the specified grammar.
    The grammar should be defined in EBNF format.
    
    This can be useful for generating structured outputs like mathematical expressions,
    programming languages, or custom domain-specific languages.
    
    Example:
        grammar = '''
            ?start: expression
            ?expression: term (("+" | "-") term)*
            ?term: factor (("*" | "/") factor)*
            ?factor: NUMBER | "-" factor | "(" expression ")"
            %import common.NUMBER
        '''
    
    Note: Grammar-based generation is currently experimental and may have performance
    limitations. It uses greedy generation to mitigate these issues.
    
    For more details and examples, see:
    https://dottxt-ai.github.io/outlines/reference/generation/cfg/
    """

    custom_generator: Optional[Any] = None
    """Set your own outlines generator object to override the default behavior."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional parameters to pass to the underlying model.
    
    Example:
        model_kwargs = {"temperature": 0.8, "seed": 42}
    """

    @model_validator(mode="after")
    def validate_environment(self) -> "ChatOutlines":
        """Validate that outlines is installed and create a model instance."""
        num_constraints = sum(
            [
                bool(self.regex),
                bool(self.type_constraints),
                bool(self.json_schema),
                bool(self.grammar),
            ]
        )
        if num_constraints > 1:
            raise ValueError(
                "Either none or exactly one of regex, type_constraints, "
                "json_schema, or grammar can be provided."
            )
        return self.build_client()

    def build_client(self) -> "ChatOutlines":
        try:
            import outlines.models as models
        except ImportError:
            raise ImportError(
                "Could not import the Outlines library. "
                "Please install it with `pip install outlines`."
            )

        def check_packages_installed(
            packages: List[Union[str, Tuple[str, str]]],
        ) -> None:
            missing_packages = [
                pkg if isinstance(pkg, str) else pkg[0]
                for pkg in packages
                if importlib.util.find_spec(pkg[1] if isinstance(pkg, tuple) else pkg)
                is None
            ]
            if missing_packages:
                raise ImportError(
                    f"Missing packages: {', '.join(missing_packages)}. "
                    "You can install them with:\n\n"
                    f"    pip install {' '.join(missing_packages)}"
                )

        if self.backend == "llamacpp":
            check_packages_installed([("llama-cpp-python", "llama_cpp")])
            if ".gguf" in self.model:
                creator, repo_name, file_name = self.model.split("/", 2)
                repo_id = f"{creator}/{repo_name}"
            else:
                raise ValueError("GGUF file_name must be provided for llama.cpp.")
            self.client = models.llamacpp(repo_id, file_name, **self.model_kwargs)
        elif self.backend == "transformers":
            check_packages_installed(["transformers", "torch", "datasets"])
            self.client = models.transformers(
                model_name=self.model, **self.model_kwargs
            )
        elif self.backend == "transformers_vision":
            if hasattr(models, "transformers_vision"):
                from transformers import LlavaNextForConditionalGeneration

                self.client = models.transformers_vision(
                    self.model,
                    model_class=LlavaNextForConditionalGeneration,
                    **self.model_kwargs,
                )
            else:
                raise ValueError("transformers_vision backend is not supported")
        elif self.backend == "vllm":
            if platform.system() == "Darwin":
                raise ValueError("vLLM backend is not supported on macOS.")
            check_packages_installed(["vllm"])
            self.client = models.vllm(self.model, **self.model_kwargs)
        elif self.backend == "mlxlm":
            check_packages_installed(["mlx"])
            self.client = models.mlxlm(self.model, **self.model_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        return self

    @property
    def _llm_type(self) -> str:
        return "outlines-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "stop_at": self.stop,
            **self.model_kwargs,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "regex": self.regex,
            "type_constraints": self.type_constraints,
            "json_schema": self.json_schema,
            "grammar": self.grammar,
            **self._default_params,
        }

    @property
    def _generator(self) -> Any:
        from outlines import generate

        if self.custom_generator:
            return self.custom_generator
        constraints = [
            self.regex,
            self.type_constraints,
            self.json_schema,
            self.grammar,
        ]

        num_constraints = sum(constraint is not None for constraint in constraints)
        if num_constraints != 1 and num_constraints != 0:
            raise ValueError(
                "Either none or exactly one of regex, type_constraints, "
                "json_schema, or grammar can be provided."
            )
        if self.regex:
            return generate.regex(self.client, regex_str=self.regex)
        if self.type_constraints:
            return generate.format(self.client, python_type=self.type_constraints)
        if self.json_schema:
            return generate.json(self.client, schema_object=self.json_schema)
        if self.grammar:
            return generate.cfg(self.client, cfg_str=self.grammar)
        return generate.text(self.client)

    def _convert_messages_to_openai_format(
        self, messages: list[BaseMessage]
    ) -> list[dict]:
        return [convert_message_to_dict(message) for message in messages]

    def _convert_messages_to_prompt(self, messages: list[BaseMessage]) -> str:
        """Convert a list of messages to a single prompt."""
        if self.backend == "llamacpp":  # get base_model_name from gguf repo_id
            from huggingface_hub import ModelCard

            repo_creator, gguf_repo_name, file_name = self.model.split("/")
            model_card = ModelCard.load(f"{repo_creator}/{gguf_repo_name}")
            if hasattr(model_card.data, "base_model"):
                model_name = model_card.data.base_model
            else:
                raise ValueError(f"Base model name not found for {self.model}")
        else:
            model_name = self.model

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name).apply_chat_template(
            self._convert_messages_to_openai_format(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Optional[Union[Dict, bool, str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model

        tool_choice: does not currently support "any", "auto" choices like OpenAI
            tool-calling API. should be a dict of the form to force this tool
            {"type": "function", "function": {"name": <<tool_name>>}}.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        tool_names = [ft["function"]["name"] for ft in formatted_tools]
        if tool_choice:
            if isinstance(tool_choice, dict):
                if not any(
                    tool_choice["function"]["name"] == name for name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, str):
                chosen = [
                    f for f in formatted_tools if f["function"]["name"] == tool_choice
                ]
                if not chosen:
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, bool):
                if len(formatted_tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be specified when a single tool is "
                        f"passed in. Received {len(tools)} tools."
                    )
                tool_choice = formatted_tools[0]

        kwargs["tool_choice"] = tool_choice
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind_tools(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        if get_origin(schema) is TypedDict:
            raise NotImplementedError("TypedDict is not supported yet by Outlines")

        self.json_schema = schema

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            parser: Union[PydanticOutputParser, JsonOutputParser] = (
                PydanticOutputParser(pydantic_object=schema)
            )
        else:
            parser = JsonOutputParser()

        if include_raw:  # TODO
            raise NotImplementedError("include_raw is not yet supported")

        return self | parser

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop_at"] = stop

        prompt = self._convert_messages_to_prompt(messages)

        response = ""
        if self.streaming:
            for chunk in self._stream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                if isinstance(chunk.message.content, str):
                    response += chunk.message.content
                else:
                    raise ValueError(
                        "Invalid content type, only str is supported, "
                        f"got {type(chunk.message.content)}"
                    )
        else:
            response = self._generator(prompt, **params)

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop_at"] = stop

        prompt = self._convert_messages_to_prompt(messages)

        for token in self._generator.stream(prompt, **params):
            if run_manager:
                run_manager.on_llm_new_token(token)
            message_chunk = AIMessageChunk(content=token)
            chunk = ChatGenerationChunk(message=message_chunk)
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if hasattr(self._generator, "agenerate"):
            params = {**self._default_params, **kwargs}
            if stop:
                params["stop_at"] = stop

            prompt = self._convert_messages_to_prompt(messages)
            response = await self._generator.agenerate(prompt, **params)

            message = AIMessage(content=response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        elif self.streaming:
            response = ""
            async for chunk in self._astream(messages, stop, run_manager, **kwargs):
                response += chunk.message.content or ""
            message = AIMessage(content=response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        else:
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if hasattr(self._generator, "astream"):
            params = {**self._default_params, **kwargs}
            if stop:
                params["stop_at"] = stop

            prompt = self._convert_messages_to_prompt(messages)

            async for token in self._generator.astream(prompt, **params):
                if run_manager:
                    await run_manager.on_llm_new_token(token)
                message_chunk = AIMessageChunk(content=token)
                chunk = ChatGenerationChunk(message=message_chunk)
                yield chunk
        else:
            async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
                yield chunk
