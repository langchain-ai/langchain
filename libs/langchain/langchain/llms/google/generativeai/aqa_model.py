from typing import Any, List

import google.ai.generativelanguage as genai
from pydantic import BaseModel, PrivateAttr

import langchain.vectorstores.google.generativeai.genai_extension as genaix


class AqaModel(BaseModel):
    """AQA model."""

    _client: genai.TextServiceClient = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = genaix.build_text_service()

    def generate_text_answer(
        self, prompt: str, passages: List[str], answer_style: genai.AnswerStyle
    ) -> genaix.TextAnswer:
        return genaix.generate_text_answer(
            prompt=prompt,
            passages=passages,
            client=self._client,
            answer_style=answer_style,
        )
