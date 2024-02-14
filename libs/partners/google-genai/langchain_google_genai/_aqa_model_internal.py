from typing import Any, List, Optional

import google.ai.generativelanguage as genai
from langchain.pydantic_v1 import BaseModel, PrivateAttr

from . import _genai_extension as genaix


class AqaModel(BaseModel):
    """AQA model."""

    _client: genai.GenerativeServiceClient = PrivateAttr()
    _answer_style: int = PrivateAttr()
    _safety_settings: List[genai.SafetySetting] = PrivateAttr()
    _temperature: Optional[float] = PrivateAttr()

    def __init__(
        self,
        answer_style: int = genai.GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
        safety_settings: List[genai.SafetySetting] = [],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = genaix.build_generative_service()
        self._answer_style = answer_style
        self._safety_settings = safety_settings
        self._temperature = temperature

    def generate_answer(
        self,
        prompt: str,
        passages: List[str],
    ) -> genaix.GroundedAnswer:
        return genaix.generate_answer(
            prompt=prompt,
            passages=passages,
            client=self._client,
            answer_style=self._answer_style,
            safety_settings=self._safety_settings,
            temperature=self._temperature,
        )
