"""Wrapper around Perplexity APIs."""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import (
    from_env,
    get_pydantic_field_names,
)
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class ChatPerplexity(BaseChatModel):
    """Modified `Perplexity AI` Chat models API with citation support.
    
    This version includes citations in the output when available from the API response.
    """

    client: Any = None  #: :meta private:
    model: str = "llama-3.1-sonar-small-128k-online"
    """Model name."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    pplx_api_key: Optional[str] = Field(
        default_factory=from_env("PPLX_API_KEY", default=None), alias="api_key"
    )
    request_timeout: Optional[Union[float, Tuple[float, float]]] = Field(
        None, alias="timeout"
    )
    max_retries: int = 6
    streaming: bool = False
    max_tokens: Optional[int] = None

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"pplx_api_key": "PPLX_API_KEY"}

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not a default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            self.client = openai.OpenAI(
                api_key=self.pplx_api_key, base_url="https://api.perplexity.ai"
            )
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling PerplexityChat API."""
        return {
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _convert_delta_to_message_chunk(
        self, _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        role = _dict.get("role")
        content = _dict.get("content") or ""
        additional_kwargs: Dict = {}
        if _dict.get("function_call"):
            function_call = dict(_dict["function_call"])
            if "name" in function_call and function_call["name"] is None:
                function_call["name"] = ""
            additional_kwargs["function_call"] = function_call
        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role == "function" or default_class == FunctionMessageChunk:
            return FunctionMessageChunk(content=content, name=_dict["name"])
        elif role == "tool" or default_class == ToolMessageChunk:
            return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)
        else:
            return default_class(content=content)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk

        if stop:
            params["stop_sequences"] = stop
        stream_resp = self.client.chat.completions.create(
            model=params["model"], messages=message_dicts, stream=True
        )
        
        # Track whether we've seen the last chunk
        final_chunk_seen = False
        citations = []
        
        for chunk in stream_resp:
            if not isinstance(chunk, dict):
                chunk_dict = chunk.dict()
            else:
                chunk_dict = chunk
                
            # Store citations if they exist
            if "citations" in chunk_dict:
                citations = chunk_dict["citations"]
                
            if len(chunk_dict["choices"]) == 0:
                continue
                
            choice = chunk_dict["choices"][0]
            
            # Check if this is the final chunk
            if choice.get("finish_reason") is not None:
                final_chunk_seen = True
                
            # Create message chunk
            chunk = self._convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            
            # Add citations if this is the final chunk and we have citations
            if final_chunk_seen and citations:
                citation_text = "\n\nCitations:\n" + "\n".join(
                    f"[{i+1}] {citation}" for i, citation in enumerate(citations)
                )
                chunk = self._convert_delta_to_message_chunk(
                    {"role": "assistant", "content": citation_text},
                    default_chunk_class
                )
            
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _append_citations(self, content: str, citations: List[str]) -> str:
        """Append citations to the message content."""
        if citations:
            content += "\n\nCitations:\n"
            for i, citation in enumerate(citations, 1):
                content += f"[{i}] {citation}\n"
        return content

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
            
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.completions.create(
            model=params["model"], messages=message_dicts
        )
        
        # Get the response content
        content = response.choices[0].message.content
        
        # Get citations if available (assuming response is converted to dict)
        response_dict = response.dict() if hasattr(response, 'dict') else response
        citations = response_dict.get('citations', [])
        
        # Append citations to content if available
        if citations:
            content = self._append_citations(content, citations)
            
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        pplx_creds: Dict[str, Any] = {
            "api_key": self.pplx_api_key,
            "api_base": "https://api.perplexity.ai",
            "model": self.model,
        }
        return {**pplx_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "perplexitychat"