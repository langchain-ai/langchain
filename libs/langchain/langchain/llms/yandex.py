from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class YandexGPT(LLM):
    """Yandex large language models.

    To use, you should have the ``yandexcloud`` python package installed.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import YandexGPT
            yandex_gpt = YandexGPT(iam_token="")
    """

    def __init__(self, iam_token: str):
        super()
        try:
            from yandexcloud import SDK
        except ImportError as e:
            raise ImportError(
                "Please install YandexCloud SDK" " with `pip install yandexcloud`."
            ) from e
        super().__init__(
            **{
                "iam_token": iam_token,
            }
        )
        self.sdk = SDK(iam_token=iam_token)

    model_name: str = "general"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    sdk: Any

    @property
    def _llm_type(self) -> str:
        return "yandex llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            from google.protobuf.wrappers_pb2 import DoubleValue, Int64Value
            from yandex.cloud.ai.llm.v1alpha.llm_pb2 import GenerationOptions
            from yandex.cloud.ai.llm.v1alpha.llm_service_pb2 import (
                InstructRequest,
                InstructResponse,
            )
            from yandex.cloud.ai.llm.v1alpha.llm_service_pb2_grpc import (
                TextGenerationAsyncServiceStub,
            )
        except ImportError as e:
            raise ImportError(
                "Please install YandexCloud SDK" " with `pip install yandexcloud`."
            ) from e

        request = InstructRequest(
            model="general",
            request_text=prompt,
            generation_options=GenerationOptions(
                temperature=DoubleValue(value=self.temperature),
                max_tokens=Int64Value(value=self.max_tokens),
            ),
        )
        operation = self.sdk.client(TextGenerationAsyncServiceStub).Instruct(request)
        res = self.sdk.wait_operation_and_get_result(
            operation, response_type=InstructResponse
        )
        return res.response.alternatives[0].text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
