from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import pre_init
from pydantic import BaseModel, Field

from langchain_community.adapters.openai import convert_message_to_dict


class ChatOutlines(BaseChatModel):
    """Chat model wrapper for the Outlines library."""

    client: Any  # :meta private:

    model_identifier: str = Field(..., alias="model")
    """Identifier for the model to use with Outlines.
    
    The model_identifier should be a string in the format "provider/model_name", where:
    
    - "provider" specifies the model type or source. Supported providers are:
      - "llamacpp": For GGUF models using llama.cpp
      - "transformers": For Hugging Face Transformers models
      - "transformers_vision": For vision-language models (e.g., LLaVA)
      - "vllm": For models using the vLLM library
      - "mlxlm": For models using the MLX framework
    
    - "model_name" is the specific model identifier, which can be:
      - A Hugging Face model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
      - A local path to a model
      - For GGUF models, the format is "repo_id/file_name"
        (e.g., "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf")
    
    Examples:
    - "llamacpp/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    - "transformers/meta-llama/Llama-2-7b-chat-hf"
    - "vllm/meta-llama/Llama-2-7b-chat-hf"
    
    Note: Ensure you have the necessary dependencies installed for the chosen provider.
    The system will attempt to import required packages and may raise an ImportError
    if they are not available."""

    max_tokens: int = 256
    """The maximum number of tokens to generate."""

    stop: Optional[List[str]] = Field(None, alias="stop_at")
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

    json: Optional[Union[Any, Dict, Callable]] = None
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

    @pre_init
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that outlines is installed and create a model instance."""
        try:
            import outlines.models as models
        except ImportError:
            raise ImportError(
                "Could not import the Outlines library. "
                "Please install it with `pip install outlines`."
            )

        model_identifier: str = values["model_identifier"]

        if "/" not in model_identifier:
            raise ValueError(f"Unsupported model identifier: {model_identifier}")

        model_name = model_identifier.split("/", 1)[1]
        if ".gguf" in model_name:
            repo_id, file_name = model_name.split("/", 1)
        else:
            repo_id = model_name
            file_name = None

        def check_packages_installed(packages: List[Union[str, Tuple[str, str]]]):
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

        if model_identifier.startswith("llamacpp/"):
            check_packages_installed([("llama_cpp", "llama-cpp-python")])
            model = models.llamacpp(repo_id, file_name, **values["model_kwargs"])
        elif model_identifier.startswith("transformers/"):
            check_packages_installed(["transformers", "torch", "datasets"])
            model = models.transformers(model_name=repo_id, **values["model_kwargs"])
        elif model_identifier.startswith("transformers_vision/"):
            check_packages_installed(
                ["transformers", "datasets", "torchvision", "PIL", "flash_attn"]
            )
            from transformers import LlavaNextForConditionalGeneration

            model = models.transformers_vision(
                repo_id,
                model_class=LlavaNextForConditionalGeneration,
                **values["model_kwargs"],
            )
        elif model_identifier.startswith("vllm/"):
            check_packages_installed(["vllm"])
            model = models.vllm(repo_id, **values["model_kwargs"])
        elif model_identifier.startswith("mlxlm/"):
            check_packages_installed(["mlx"])
            model = models.mlxlm(repo_id, **values["model_kwargs"])
        else:
            # todo add dynamic handler that just tries to load and maybe fails
            raise ValueError(f"Unsupported model identifier: {model_identifier}")
        values["client"] = model
        return values

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
            "model_identifier": self.model_identifier,
            "regex": self.regex,
            "type_constraints": self.type_constraints,
            "json": self.json,
            "grammar": self.grammar,
            **self._default_params,
        }

    @property
    def _generator(self) -> Any:
        from outlines import generate

        if self.custom_generator:
            return self.custom_generator
        if (
            sum(
                [
                    self.regex is not None,
                    self.type_constraints is not None,
                    self.json is not None,
                    self.grammar is not None,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Only one of regex, type_constraints, json, or grammar can be provided."
            )
        if self.regex:
            return generate.regex(self.client, regex=self.regex)
        if self.type_constraints:
            return generate.format(self.client, python_type=self.type_constraints)
        if self.json:
            return generate.json(self.client, schema_object=self.json)
        if self.grammar:
            return generate.cfg(self.client, grammar=self.grammar)
        return generate.text(self.client)

    @property
    def tokenizer(self) -> Any:
        """Access the tokenizer for the underlying model.

        .encode() to tokenize text.
        .decode() to convert tokens back to text.
        """
        if hasattr(self.client, "tokenizer"):
            return self.client.tokenizer
        raise ValueError("Tokenizer not found")

    def _convert_messages_to_openai_format(
        self, messages: list[BaseMessage]
    ) -> list[dict]:
        return [convert_message_to_dict(message) for message in messages]

    def _convert_messages_to_prompt(self, messages: list[BaseMessage]) -> str:
        """Convert a list of messages to a single prompt."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, self.tools, self.documents
            )
        provider, model_name = self.model_identifier.split("/", 1)

        if provider == "llamacpp":  # get base_model_name from gguf repo_id
            from huggingface_hub import ModelCard

            repo_creator, gguf_repo_name, file_name = model_name.split("/")
            model_card = ModelCard.load(f"{repo_creator}/{gguf_repo_name}")
            if hasattr(model_card.data, "base_model"):
                model_name = model_card.data.base_model
            else:
                raise ValueError(f"Base model name not found for {model_name}")

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name).apply_chat_template(
            self._convert_messages_to_openai_format(messages),
            self.tools,
            self.documents,
        )

    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[
        PromptValue
        | str
        | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]],
        BaseMessage,
    ]:
        return super().bind_tools(tools, **kwargs)
        # todo implement

    def with_structured_output(
        self,
        schema: Dict | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[
        PromptValue
        | str
        | Sequence[BaseMessage | List[str] | Tuple[str] | str | Dict[str, Any]],
        Dict | BaseModel,
    ]:
        return super().with_structured_output(schema, include_raw=include_raw, **kwargs)
        # todo implement

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
        print("CHECK PROMPT", prompt[:50])

        response = ""
        if self.streaming:
            for chunk in self._stream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                response += chunk.message.content
        else:
            response = self._generator(prompt, **params)

        message = AIMessage(content=response)
        # todo maybe add better conversion thatn just ai message
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
            # todo maybe add better conversion thatn just ai message chunk
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
                response += chunk.message.content
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
