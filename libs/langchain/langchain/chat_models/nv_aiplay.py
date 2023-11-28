"""Chat Model Components Derived from ChatModel/NVAIPlay"""

from typing import Sequence, Union

from langchain.llms.nv_aiplay import NVAIPlayBaseModel
from langchain.pydantic_v1 import Field

## NOTE: This file should not be ran in isolation as a single-file standalone.
## Please use llms.nv_aiplay instead.
from .base import SimpleChatModel


class NVAIPlayChat(NVAIPlayBaseModel, SimpleChatModel):
    def __init__(
        self,
        model_name: str = "",
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        streaming: bool = False,
        stop: Union[Sequence[str], str] = [],
        **kwargs: dict,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            streaming=streaming,
            **kwargs,
        )


################################################################################


class LlamaChat(NVAIPlayChat):
    def __init__(
        self,
        model_name: str = "llama2_13b",
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        streaming: bool = False,
        stop: Union[Sequence[str], str] = [],
        **kwargs: dict,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            streaming=streaming,
            **kwargs,
        )

class MistralChat(NVAIPlayChat):
    def __init__(
        self,
        model_name: str = "mistral",
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        streaming: bool = False,
        stop: Union[Sequence[str], str] = [],
        **kwargs: dict,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            streaming=streaming,
            **kwargs,
        )

class SteerLMChat(NVAIPlayChat):
    def __init__(
        self,
        model_name: str = "gpt_steerlm_8b",
        labels: dict = {
            "creativity": 5,
            "helpfulness": 5,
            "humor": 5,
            "quality": 5,
        },
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        streaming: bool = False,
        stop: Union[Sequence[str], str] = [],
        **kwargs: dict,
    ):
        super().__init__(
            model_name=model_name,
            labels=labels,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            streaming=streaming,
            **kwargs,
        )

class NemotronQAChat(NVAIPlayChat):
    def __init__(
        self,
        model_name: str = "gpt_qa_8b",
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        streaming: bool = False,
        stop: Union[Sequence[str], str] = [],
        **kwargs: dict,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            streaming=streaming,
            **kwargs,
        )