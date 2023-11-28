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
            model_name=Field(model_name),
            temperature=Field(temperature, le=1, gt=0),
            top_p=Field(top_p, le=1, gt=0),
            max_tokens=Field(max_tokens, le=1024, ge=32),
            stop=Field(stop),
            streaming=Field(streaming),
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
            model_name=Field(model_name),
            temperature=Field(temperature, le=1, gt=0),
            top_p=Field(top_p, le=1, gt=0),
            max_tokens=Field(max_tokens, le=1024, ge=32),
            stop=Field(stop),
            streaming=Field(streaming),
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
            model_name=Field(model_name),
            temperature=Field(temperature, le=1, gt=0),
            top_p=Field(top_p, le=1, gt=0),
            max_tokens=Field(max_tokens, le=1024, ge=32),
            stop=Field(stop),
            streaming=Field(streaming),
            **kwargs,
        )


class SteerLM(NVAIPlayChat):
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
            model_name=Field(model_name),
            labels=Field(labels),
            temperature=Field(temperature, le=1, gt=0),
            top_p=Field(top_p, le=1, gt=0),
            max_tokens=Field(max_tokens, le=1024, ge=32),
            stop=Field(stop),
            streaming=Field(streaming),
            **kwargs,
        )


class NemotronQA(NVAIPlayChat):
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
            model_name=Field(model_name),
            temperature=Field(temperature, le=1, gt=0),
            top_p=Field(top_p, le=1, gt=0),
            max_tokens=Field(max_tokens, le=1024, ge=32),
            stop=Field(stop),
            streaming=Field(streaming),
            **kwargs,
        )
