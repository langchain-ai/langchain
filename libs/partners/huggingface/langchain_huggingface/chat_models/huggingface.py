"""Hugging Face Chat Wrapper."""

from dataclasses import dataclass
from inspect import signature
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import model_validator
from typing_extensions import Self

from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""

LLM_INFERENCE_CLIENT_MAP: Final[dict[str, str]] = {"max_new_tokens": "max_tokens"}
LLM_INFERENCE_CLIENT_EXCLUDED_KWARGS: Final[set[str]] = {
    "top_k",
    "typical_p",
    "repetition_penalty",
    "return_full_text",
    "truncate",
    "stop_sequences",
    "do_sample",
    "watermark",
}


@dataclass
class TGI_RESPONSE:
    """Response from the TextGenInference API."""

    choices: List[Any]
    usage: Dict


@dataclass
class TGI_MESSAGE:
    """Message to send to the TextGenInference API."""

    role: str
    content: str
    tool_calls: List[Dict]


def _convert_message_to_chat_message(
    message: BaseMessage,
) -> Dict:
    if isinstance(message, ChatMessage):
        return dict(role=message.role, content=message.content)
    elif isinstance(message, HumanMessage):
        return dict(role="user", content=message.content)
    elif isinstance(message, AIMessage):
        if "tool_calls" in message.additional_kwargs:
            tool_calls = [
                {
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                }
                for tc in message.additional_kwargs["tool_calls"]
            ]
        else:
            tool_calls = None
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls,
        }
    elif isinstance(message, SystemMessage):
        return dict(role="system", content=message.content)
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")


def _convert_TGI_message_to_LC_message(
    _message: TGI_MESSAGE,
) -> BaseMessage:
    role = _message.role
    assert role == "assistant", f"Expected role to be 'assistant', got {role}"
    content = cast(str, _message.content)
    if content is None:
        content = ""
    additional_kwargs: Dict = {}
    if tool_calls := _message.tool_calls:
        if "arguments" in tool_calls[0]["function"]:
            functions_string = str(tool_calls[0]["function"].pop("arguments"))
            corrected_functions = functions_string.replace("'", '"')
            tool_calls[0]["function"]["arguments"] = corrected_functions
        additional_kwargs["tool_calls"] = tool_calls
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


def _convert_hf_kwargs_llm_to_chat_completions(
    **kwargs: Any,
) -> Any:
    """
    Convert a set of keyword arguments to be compatible with [InferenceClient.chat_completion](https://huggingface.co/docs/huggingface_hub/v0.26.2/en/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion)
    Overrides mapped keyword arguments and omits incompatible keyword arguments.
    """
    from huggingface_hub import InferenceClient  # type: ignore[import-untyped]

    for k, v in LLM_INFERENCE_CLIENT_MAP.items():
        if k in kwargs:
            kwargs[v] = kwargs.pop(k)

    sig = set(signature(InferenceClient.chat_completion).parameters)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig}
    invalid_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in sig and k not in LLM_INFERENCE_CLIENT_EXCLUDED_KWARGS
    }
    if invalid_kwargs:
        raise ValueError(
            "Could not convert arguments to HF format", list(invalid_kwargs.keys())
        )
    return valid_kwargs


def _is_huggingface_hub(llm: Any) -> bool:
    try:
        from langchain_community.llms.huggingface_hub import (  # type: ignore[import-not-found]
            HuggingFaceHub,
        )

        return isinstance(llm, HuggingFaceHub)
    except ImportError:
        # if no langchain community, it is not a HuggingFaceHub
        return False


def _is_huggingface_textgen_inference(llm: Any) -> bool:
    try:
        from langchain_community.llms.huggingface_text_gen_inference import (  # type: ignore[import-not-found]
            HuggingFaceTextGenInference,
        )

        return isinstance(llm, HuggingFaceTextGenInference)
    except ImportError:
        # if no langchain community, it is not a HuggingFaceTextGenInference
        return False


def _is_huggingface_endpoint(llm: Any) -> bool:
    return isinstance(llm, HuggingFaceEndpoint)


def _is_huggingface_pipeline(llm: Any) -> bool:
    return isinstance(llm, HuggingFacePipeline)


class ChatHuggingFace(BaseChatModel):
    """Hugging Face LLM's as ChatModels.

    Works with `HuggingFaceTextGenInference`, `HuggingFaceEndpoint`,
    `HuggingFaceHub`, and `HuggingFacePipeline` LLMs.

    Upon instantiating this class, the model_id is resolved from the url
    provided to the LLM, and the appropriate tokenizer is loaded from
    the HuggingFace Hub.

    Setup:
        Install ``langchain-huggingface`` and ensure your Hugging Face token
        is saved.

        .. code-block:: bash

            pip install langchain-huggingface

        .. code-block:: python

            from huggingface_hub import login
            login() # You will be prompted for your HF key, which will then be saved locally

    Key init args — completion params:
        llm: `HuggingFaceTextGenInference`, `HuggingFaceEndpoint`, `HuggingFaceHub`, or
            'HuggingFacePipeline' LLM to be used.

    Key init args — client params:
        custom_get_token_ids: Optional[Callable[[str], List[int]]]
            Optional encoder to use for counting tokens.
        metadata: Optional[Dict[str, Any]]
            Metadata to add to the run trace.
        tags: Optional[List[str]]
            Tags to add to the run trace.
        tokenizer: Any
        verbose: bool
            Whether to print out response text.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEndpoint,
            ChatHuggingFace

            llm = HuggingFaceEndpoint(
                repo_id="microsoft/Phi-3-mini-4k-instruct",
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            )

            chat = ChatHuggingFace(llm=llm, verbose=True)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user
                sentence to French."),
                ("human", "I love programming."),
            ]

            chat(...).invoke(messages)

        .. code-block:: python

            AIMessage(content='Je ai une passion pour le programme.\n\nIn
            French, we use "ai" for masculine subjects and "a" for feminine
            subjects. Since "programming" is gender-neutral in English, we
            will go with the masculine "programme".\n\nConfirmation: "J\'aime
            le programme." is more commonly used. The sentence above is
            technically accurate, but less commonly used in spoken French as
            "ai" is used less frequently in everyday speech.',
            response_metadata={'token_usage': ChatCompletionOutputUsage
            (completion_tokens=100, prompt_tokens=55, total_tokens=155),
            'model': '', 'finish_reason': 'length'},
            id='run-874c24b7-0272-4c99-b259-5d6d7facbc56-0')

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='Je ai une passion pour le programme.\n\nIn French, we use
            "ai" for masculine subjects and "a" for feminine subjects.
            Since "programming" is gender-neutral in English,
            we will go with the masculine "programme".\n\nConfirmation:
            "J\'aime le programme." is more commonly used. The sentence
            above is technically accurate, but less commonly used in spoken
            French as "ai" is used less frequently in everyday speech.'
            response_metadata={'token_usage': ChatCompletionOutputUsage
            (completion_tokens=100, prompt_tokens=55, total_tokens=155),
            'model': '', 'finish_reason': 'length'}
            id='run-7d7b1967-9612-4f9a-911a-b2b5ca85046a-0'

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

        .. code-block:: python

            AIMessage(content='Je déaime le programming.\n\nLittérale : Je
            (j\'aime) déaime (le) programming.\n\nNote: "Programming" in
            French is "programmation". But here, I used "programming" instead
            of "programmation" because the user said "I love programming"
            instead of "I love programming (in French)", which would be
            "J\'aime la programmation". By translating the sentence
            literally, I preserved the original meaning of the user\'s
            sentence.', id='run-fd850318-e299-4735-b4c6-3496dc930b1d-0')

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state,
                e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state,
                e.g. San Francisco, CA")

            chat_with_tools = chat.bind_tools([GetWeather, GetPopulation])
            ai_msg = chat_with_tools.invoke("Which city is hotter today and
            which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': '0'}]

    Response metadata
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python
            {'token_usage': ChatCompletionOutputUsage(completion_tokens=100,
            prompt_tokens=8, total_tokens=108),
             'model': '',
             'finish_reason': 'length'}

    """  # noqa: E501

    llm: Any
    """LLM, must be of type HuggingFaceTextGenInference, HuggingFaceEndpoint,
        HuggingFaceHub, or HuggingFacePipeline."""
    # TODO: Is system_message used anywhere?
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_id: Optional[str] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        from transformers import AutoTokenizer  # type: ignore[import]

        self._resolve_model_id()

        self.tokenizer = (
            AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer is None
            else self.tokenizer
        )

    @model_validator(mode="after")
    def validate_llm(self) -> Self:
        if (
            not _is_huggingface_hub(self.llm)
            and not _is_huggingface_textgen_inference(self.llm)
            and not _is_huggingface_endpoint(self.llm)
            and not _is_huggingface_pipeline(self.llm)
        ):
            raise TypeError(
                "Expected llm to be one of HuggingFaceTextGenInference, "
                "HuggingFaceEndpoint, HuggingFaceHub, HuggingFacePipeline "
                f"received {type(self.llm)}"
            )
        return self

    def _create_chat_result(self, response: TGI_RESPONSE) -> ChatResult:
        generations = []
        finish_reason = response.choices[0].finish_reason
        gen = ChatGeneration(
            message=_convert_TGI_message_to_LC_message(response.choices[0].message),
            generation_info={"finish_reason": finish_reason},
        )
        generations.append(gen)
        token_usage = response.usage
        model_object = self.llm.inference_server_url
        llm_output = {"token_usage": token_usage, "model": model_object}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _invocation_params(self, stop: Optional[List[str]], **kwargs: Any) -> Any:
        """
        Convert to params from the inference endpoint config
        to the `/chat_completions` format to preserve endpoint config.
        https://github.com/langchain-ai/langchain/issues/23586
        """
        llm_kwargs = self.llm._invocation_params(stop, **kwargs)
        chat_completion_kwargs = _convert_hf_kwargs_llm_to_chat_completions(
            **llm_kwargs
        )
        return chat_completion_kwargs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if _is_huggingface_textgen_inference(self.llm):
            message_dicts = self._create_message_dicts(messages, stop)
            answer = self.llm.client.chat(messages=message_dicts, **kwargs)
            return self._create_chat_result(answer)
        elif _is_huggingface_endpoint(self.llm):
            message_dicts = self._create_message_dicts(messages, stop)
            kwargs = self._invocation_params(stop, **kwargs)
            answer = self.llm.client.chat_completion(messages=message_dicts, **kwargs)
            return self._create_chat_result(answer)
        else:
            llm_input = self._to_chat_prompt(messages)
            llm_result = self.llm._generate(
                prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
            )
            return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if _is_huggingface_textgen_inference(self.llm):
            message_dicts = self._create_message_dicts(messages, stop)
            answer = await self.llm.async_client.chat(messages=message_dicts, **kwargs)
            return self._create_chat_result(answer)
        else:
            llm_input = self._to_chat_prompt(messages)
            llm_result = await self.llm._agenerate(
                prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
            )
            return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    def _resolve_model_id(self) -> None:
        """Resolve the model_id from the LLM's inference_server_url"""

        from huggingface_hub import list_inference_endpoints  # type: ignore[import]

        if _is_huggingface_hub(self.llm) or (
            hasattr(self.llm, "repo_id") and self.llm.repo_id
        ):
            self.model_id = self.llm.repo_id
            return
        elif _is_huggingface_textgen_inference(self.llm):
            endpoint_url: Optional[str] = self.llm.inference_server_url
        elif _is_huggingface_pipeline(self.llm):
            self.model_id = self.llm.model_id
            return
        else:
            endpoint_url = self.llm.endpoint_url
        available_endpoints = list_inference_endpoints("*")
        for endpoint in available_endpoints:
            if endpoint.url == endpoint_url:
                self.model_id = endpoint.repository

        if not self.model_id:
            raise ValueError(
                "Failed to resolve model_id:"
                f"Could not find model id for inference server: {endpoint_url}"
                "Make sure that your Hugging Face token has access to the endpoint."
            )

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]["function"]["name"]
                    != tool_choice["function"]["name"]
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> List[Dict[Any, Any]]:
        message_dicts = [_convert_message_to_chat_message(m) for m in messages]
        return message_dicts

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat-wrapper"
