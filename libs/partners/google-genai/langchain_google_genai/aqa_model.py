"""Google GenerativeAI Attributed Question and Answering (AQA) service.

The GenAI Semantic AQA API is a managed end to end service that allows
developers to create responses grounded on specified passages based on
a user query. For more information visit:
https://developers.generativeai.google/guide
"""

from typing import Any, List, Optional

from langchain.pydantic_v1 import BaseModel, PrivateAttr
from langchain.schema.runnable import RunnableSerializable
from langchain.schema.runnable.config import RunnableConfig

_import_err_msg = (
    "`google.generativeai` package not found, "
    "please run `pip install google-generativeai`"
)


class AqaModelInput(BaseModel):
    """Input to `GenAIAqa.invoke`.

    Attributes:
        prompt: The user's inquiry.
        source_passages: A list of passage that the LLM should use only to
            answer the user's inquiry.
    """

    prompt: str
    source_passages: List[str]


class AqaModelOutput(BaseModel):
    """Output from `GenAIAqa.invoke`.

    Attributes:
        answer: The answer to the user's inquiry.
        attributed_passages: A list of passages that the LLM used to construct
            the answer.
        answerable_probability: The probability of the question being answered
            from the provided passages.
    """

    answer: str
    attributed_passages: List[str]
    answerable_probability: float


class GenAIAqa(RunnableSerializable[AqaModelInput, AqaModelOutput]):
    """Google's Attributed Question and Answering service.

    Given a user's query and a list of passages, Google's server will return
    a response that is grounded to the provided list of passages. It will not
    base the response on parametric memory.

    Attributes:
        answer_style: keyword-only argument. See
            `google.ai.generativelanguage.AnswerStyle` for details.
    """

    # Actual type is .aqa_model.AqaModel.
    _client: Any = PrivateAttr()

    # Actual type is genai.AnswerStyle.
    # 1 = ABSTRACTIVE.
    # Cannot use the actual type here because user may not have
    # google.generativeai installed.
    answer_style: int = 1

    def __init__(self, **kwargs: Any) -> None:
        """Construct a Google Generative AI AQA model.

        All arguments are optional.

        Args:
            answer_style: See
              `google.ai.generativelanguage.GenerateAnswerRequest.AnswerStyle`.
            safety_settings: See `google.ai.generativelanguage.SafetySetting`.
            temperature: 0.0 to 1.0.
        """
        try:
            from ._aqa_model_internal import AqaModel
        except ImportError:
            raise ImportError(_import_err_msg)

        super().__init__(**kwargs)
        self._client = AqaModel(**kwargs)

    def invoke(
        self, input: AqaModelInput, config: Optional[RunnableConfig] = None
    ) -> AqaModelOutput:
        """Generates a grounded response using the provided passages."""
        try:
            from ._aqa_model_internal import AqaModel
        except ImportError:
            raise ImportError(_import_err_msg)

        client: AqaModel = self._client

        response = client.generate_answer(
            prompt=input.prompt, passages=input.source_passages
        )

        return AqaModelOutput(
            answer=response.answer,
            attributed_passages=[
                passage.text for passage in response.attributed_passages
            ],
            answerable_probability=response.answerable_probability or 0.0,
        )
