from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.pydantic_v1 import Field, root_validator

from langchain_openai.llms.base import BaseOpenAI


class VLLMOpenAI(BaseOpenAI):
    """VLLM large language models.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_openai import VLLMOpenAI

            model = VLLMOpenAI(model_name="gpt-3.5-turbo-instruct")
    """

    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L338
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
    """Controls the stopping condition for beam search. It accepts the following 
    values: `True`, where the generation stops as soon as there are `best_of`
    complete candidates; `False`, where an heuristic is applied and the generation
    stops when is it very unlikely to find better candidates; 
    `"never"`, where the beam search procedure only stops when there 
    cannot be better candidates (canonical beam search algorithm)."""
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    """List of tokens that stop the generation when they are generated. 
    The returned output will contain the stop tokens 
    unless the stop tokens are special tokens."""
    ignore_eos: Optional[bool] = False
    """Whether to ignore the EOS token and continue generating
    tokens after the EOS token is generated."""
    min_tokens: int = 0
    """Minimum number of tokens to generate per output sequence before 
    EOS or stop_token_ids can be generated."""
    skip_special_tokens: Optional[bool] = True
    """Whether to skip special tokens in the output."""
    spaces_between_special_tokens: Optional[bool] = True
    """Whether to add spaces between special tokens in the output."""
    truncate_prompt_tokens: Optional[int] = None
    """If set to an integer k, will use only the last k tokens from the prompt."""
    include_stop_str_in_output: Optional[bool] = False
    """Whether to include the stop string in the output.
    This is only applied when the stop or stop_token_ids is set."""
    guided_json: Optional[Union[str, Dict]] = None
    """If specified, the output will follow the JSON schema."""
    guided_regex: Optional[str] = None
    """If specified, the output will follow the regex pattern."""
    guided_choice: Optional[List[str]] = None
    """If specified, the output will be exactly one of the choices."""
    guided_grammar: Optional[str] = None
    """If specified, the output will follow the context free grammar."""
    guided_decoding_backend: Optional[Literal["outlines", "lm-format-enforcer"]] = None
    """If specified, will override the default guided decoding backend of the server
    for this specific request."""
    guided_whitespace_pattern: Optional[str] = None
    """If specified, will override the default whitespace pattern
    for guided json decoding."""

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
            "truncate_prompt_tokens": self.truncate_prompt_tokens,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "guided_json": self.guided_json,
            "guided_regex": self.guided_regex,
            "guided_choice": self.guided_choice,
            "guided_grammar": self.guided_grammar,
            "guided_decoding_backend": self.guided_decoding_backend,
            "guided_whitespace_pattern": self.guided_whitespace_pattern,
            **(self.extra_body or {}),
        }

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "openai"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.openai_api_base:
            attributes["openai_api_base"] = self.openai_api_base

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.openai_proxy:
            attributes["openai_proxy"] = self.openai_proxy

        return attributes
