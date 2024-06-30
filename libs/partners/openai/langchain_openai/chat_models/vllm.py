from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from langchain_core.outputs import ChatGenerationChunk
from langchain_core.pydantic_v1 import Field, root_validator

from langchain_openai.chat_models.base import BaseChatOpenAI


class VLLMChatOpenAI(BaseChatOpenAI):
    """VLLMOpenAI chat model integration."""

    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    use_beam_search: Optional[bool] = False
    """Whether to use beam search instead of sampling."""
    top_k: int = -1
    """Integer that controls the number of top tokens to consider.
    Set to -1 to consider all tokens"""
    min_p: float = 0.0
    """Float that represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token."""
    repetition_penalty: float = 1.0
    """Float that penalizes new tokens based on whether they appear in the prompt
    and the generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""
    length_penalty: Optional[float] = 1.0
    """Float that penalizes sequences based on their length. Used in beam search"""
    early_stopping: Optional[bool] = False
    """Controls the stopping condition for beam search. 
    It accepts the following values: `True`, where the generation stops as soon 
    as there are `best_of` complete candidates; `False`, where an heuristic is applied
    and the generation stops when is it very unlikely to find better candidates;
    `"never"`, where the beam search procedure only stops when there cannot be better
    candidates (canonical beam search algorithm)."""
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    """List of tokens that stop the generation when they are generated. 
    The returned output will contain the stop tokens unless
    the stop tokens are special tokens."""
    ignore_eos: Optional[bool] = False
    """Whether to ignore the EOS token and continue generating tokens 
    after the EOS token is generated."""
    min_tokens: int = 0
    """Minimum number of tokens to generate per output sequence."""
    skip_special_tokens: Optional[bool] = True
    """Minimum number of tokens to generate per output sequence before EOS
    or stop_token_ids can be generated."""
    spaces_between_special_tokens: Optional[bool] = True
    """Whether to add spaces between special tokens in the output."""
    echo: bool = False
    """If true, the new message will be prepended with the last message
    if they belong to the same role."""
    add_generation_prompt: bool = True
    """If true, the generation prompt will be added to the chat template. 
    This is a parameter used by chat template in tokenizer config of the model."""
    add_special_tokens: bool = False
    """If true, special tokens (e.g. BOS) will be added to the prompt kon top of 
    what is added by the chat template. For most models, the chat template takes 
    care of adding the special tokens so this should be set to False."""
    include_stop_str_in_output: Optional[bool] = False
    """Whether to include the stop string in the output. This is only applied when
    the stop or stop_token_ids is set."""
    guided_json: Optional[Union[str, Dict]] = None
    """If specified, the output will follow the JSON schema."""
    guided_regex: Optional[str] = None
    """If specified, the output will follow the regex pattern."""
    guided_choice: Optional[List[str]] = None
    """If specified, the output will be exactly one of the choices."""
    guided_grammar: Optional[str] = None
    """If specified, the output will follow the context free grammar."""
    guided_decoding_backend: Optional[Literal["outlines", "lm-format-enforcer"]] = None
    """If specified, will override the default guided decoding backend 
    of the server for this specific request. 
    If set, must be either 'outlines' / 'lm-format-enforcer'"""
    guided_whitespace_pattern: Optional[str] = None
    """If specified, will override the default 
    whitespace pattern for guided json decoding."""

    @root_validator(pre=True)
    def check_params(cls, values: Dict) -> Dict:
        guide_count = sum(
            [
                "guided_json" in values and values["guided_json"] is not None,
                "guided_regex" in values and values["guided_regex"] is not None,
                "guided_choice" in values and values["guided_choice"] is not None,
            ]
        )
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "(`guided_json`, `guided_regex` or `guided_choice`)."
            )
        if values.get("stream_options") and not values.get("stream"):
            raise ValueError("Stream options can only be defined when stream is True.")
        if values.get("min_p") and (values["min_p"] < 0 or values["min_p"] > 1):
            raise ValueError("`min_p` must be between 0 and 1")
        if (
            values.get("truncate_prompt_tokens")
            and values["truncate_prompt_tokens"] < 1
        ):
            raise ValueError("`truncate_prompt_tokens` must be greater than 1")

        return values

    @property
    def _extra_body(self) -> Dict[str, Any]:
        return {
            "best_of": self.best_of,
            "use_beam_search": self.use_beam_search,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "stop_token_ids": self.stop_token_ids,
            "ignore_eos": self.ignore_eos,
            "min_tokens": self.min_tokens,
            "skip_special_tokens": self.skip_special_tokens,
            "spaces_between_special_tokens": self.spaces_between_special_tokens,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "guided_json": self.guided_json,
            "guided_regex": self.guided_regex,
            "guided_choice": self.guided_choice,
            "guided_grammar": self.guided_grammar,
            "guided_decoding_backend": self.guided_decoding_backend,
            "guided_whitespace_pattern": self.guided_whitespace_pattern,
            "echo": self.echo,
            **(self.extra_body or {}),
        }

    stream_usage: bool = False
    """Whether to include usage metadata in streaming output. If True, additional
    message chunks will be generated during the stream including usage metadata.
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "openai"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.openai_api_base:
            attributes["openai_api_base"] = self.openai_api_base

        if self.openai_proxy:
            attributes["openai_proxy"] = self.openai_proxy

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    def _should_stream_usage(
        self, stream_usage: Optional[bool] = None, **kwargs: Any
    ) -> bool:
        """Determine whether to include usage metadata in streaming output.

        For backwards compatibility, we check for `stream_options` passed
        explicitly to kwargs or in the model_kwargs and override self.stream_usage.
        """
        stream_usage_sources = [  # order of preference
            stream_usage,
            kwargs.get("stream_options", {}).get("include_usage"),
            self.model_kwargs.get("stream_options", {}).get("include_usage"),
            self.stream_usage,
        ]
        for source in stream_usage_sources:
            if isinstance(source, bool):
                return source
        return self.stream_usage

    def _stream(
        self, *args: Any, stream_usage: Optional[bool] = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        """Set default stream_options."""
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        # Note: stream_options is not a valid parameter for Azure OpenAI.
        # To support users proxying Azure through ChatOpenAI, here we only specify
        # stream_options if include_usage is set to True.
        # See https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new
        # for release notes.
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}

        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, stream_usage: Optional[bool] = None, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Set default stream_options."""
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}

        async for chunk in super()._astream(*args, **kwargs):
            yield chunk
