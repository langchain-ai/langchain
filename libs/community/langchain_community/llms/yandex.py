from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import SecretStr
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class _BaseYandexGPT(Serializable):
    iam_token: SecretStr = ""  # type: ignore[assignment]
    """Yandex Cloud IAM token for service or user account
    with the `ai.languageModels.user` role"""
    api_key: SecretStr = ""  # type: ignore[assignment]
    """Yandex Cloud Api Key for service account
    with the `ai.languageModels.user` role"""
    folder_id: str = ""
    """Yandex Cloud folder ID"""
    model_uri: str = ""
    """Model uri to use."""
    model_name: str = "yandexgpt-lite"
    """Model name to use."""
    model_version: str = "latest"
    """Model version to use."""
    temperature: float = 0.6
    """What sampling temperature to use.
    Should be a double number between 0 (inclusive) and 1 (inclusive)."""
    max_tokens: int = 7400
    """Sets the maximum limit on the total number of tokens
    used for both the input prompt and the generated response.
    Must be greater than zero and not exceed 7400 tokens."""
    stop: Optional[List[str]] = None
    """Sequences when completion generation will stop."""
    url: str = "llm.api.cloud.yandex.net:443"
    """The url of the API."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    sleep_interval: float = 1.0
    """Delay between API requests"""
    disable_request_logging: bool = False
    """YandexGPT API logs all request data by default. 
    If you provide personal data, confidential information, disable logging."""
    grpc_metadata: Optional[Sequence] = None

    @property
    def _llm_type(self) -> str:
        return "yandex_gpt"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_uri": self.model_uri,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "max_retries": self.max_retries,
        }

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that iam token exists in environment."""

        iam_token = convert_to_secret_str(
            get_from_dict_or_env(values, "iam_token", "YC_IAM_TOKEN", "")
        )
        values["iam_token"] = iam_token
        api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "api_key", "YC_API_KEY", "")
        )
        values["api_key"] = api_key
        folder_id = get_from_dict_or_env(values, "folder_id", "YC_FOLDER_ID", "")
        values["folder_id"] = folder_id
        if api_key.get_secret_value() == "" and iam_token.get_secret_value() == "":
            raise ValueError("Either 'YC_API_KEY' or 'YC_IAM_TOKEN' must be provided.")

        if values["iam_token"]:
            values["grpc_metadata"] = [
                ("authorization", f"Bearer {values['iam_token'].get_secret_value()}")
            ]
            if values["folder_id"]:
                values["grpc_metadata"].append(("x-folder-id", values["folder_id"]))
        else:
            values["grpc_metadata"] = [
                ("authorization", f"Api-Key {values['api_key'].get_secret_value()}"),
            ]
        if values["model_uri"] == "" and values["folder_id"] == "":
            raise ValueError("Either 'model_uri' or 'folder_id' must be provided.")
        if not values["model_uri"]:
            values["model_uri"] = (
                f"gpt://{values['folder_id']}/{values['model_name']}/{values['model_version']}"
            )
        if values["disable_request_logging"]:
            values["grpc_metadata"].append(
                (
                    "x-data-logging-enabled",
                    "false",
                )
            )
        return values


class YandexGPT(_BaseYandexGPT, LLM):
    """Yandex large language models.

    To use, you should have the ``yandexcloud`` python package installed.

    There are two authentication options for the service account
    with the ``ai.languageModels.user`` role:
        - You can specify the token in a constructor parameter `iam_token`
        or in an environment variable `YC_IAM_TOKEN`.
        - You can specify the key in a constructor parameter `api_key`
        or in an environment variable `YC_API_KEY`.

    To use the default model specify the folder ID in a parameter `folder_id`
    or in an environment variable `YC_FOLDER_ID`.

    Or specify the model URI in a constructor parameter `model_uri`

    Example:
        .. code-block:: python

            from langchain_community.llms import YandexGPT
            yandex_gpt = YandexGPT(iam_token="t1.9eu...", folder_id="b1g...")
    """

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Yandex GPT model and return the output.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = YandexGPT("Tell me a joke.")
        """
        text = completion_with_retry(self, prompt=prompt)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call the Yandex GPT model and return the output.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        text = await acompletion_with_retry(self, prompt=prompt)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text


def _make_request(
    self: YandexGPT,
    prompt: str,
) -> str:
    try:
        import grpc
        from google.protobuf.wrappers_pb2 import DoubleValue, Int64Value

        try:
            from yandex.cloud.ai.foundation_models.v1.text_common_pb2 import (
                CompletionOptions,
                Message,
            )
            from yandex.cloud.ai.foundation_models.v1.text_generation.text_generation_service_pb2 import (  # noqa: E501
                CompletionRequest,
            )
            from yandex.cloud.ai.foundation_models.v1.text_generation.text_generation_service_pb2_grpc import (  # noqa: E501
                TextGenerationServiceStub,
            )
        except ModuleNotFoundError:
            from yandex.cloud.ai.foundation_models.v1.foundation_models_pb2 import (
                CompletionOptions,
                Message,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2 import (  # noqa: E501
                CompletionRequest,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import (  # noqa: E501
                TextGenerationServiceStub,
            )
    except ImportError as e:
        raise ImportError(
            "Please install YandexCloud SDK  with `pip install yandexcloud` \
            or upgrade it to recent version."
        ) from e
    channel_credentials = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel(self.url, channel_credentials)
    request = CompletionRequest(
        model_uri=self.model_uri,
        completion_options=CompletionOptions(
            temperature=DoubleValue(value=self.temperature),
            max_tokens=Int64Value(value=self.max_tokens),
        ),
        messages=[Message(role="user", text=prompt)],
    )
    stub = TextGenerationServiceStub(channel)
    res = stub.Completion(request, metadata=self.grpc_metadata)  # type: ignore[attr-defined]
    return list(res)[0].alternatives[0].message.text


async def _amake_request(self: YandexGPT, prompt: str) -> str:
    try:
        import asyncio

        import grpc
        from google.protobuf.wrappers_pb2 import DoubleValue, Int64Value

        try:
            from yandex.cloud.ai.foundation_models.v1.text_common_pb2 import (
                CompletionOptions,
                Message,
            )
            from yandex.cloud.ai.foundation_models.v1.text_generation.text_generation_service_pb2 import (  # noqa: E501
                CompletionRequest,
                CompletionResponse,
            )
            from yandex.cloud.ai.foundation_models.v1.text_generation.text_generation_service_pb2_grpc import (  # noqa: E501
                TextGenerationAsyncServiceStub,
            )
        except ModuleNotFoundError:
            from yandex.cloud.ai.foundation_models.v1.foundation_models_pb2 import (
                CompletionOptions,
                Message,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2 import (  # noqa: E501
                CompletionRequest,
                CompletionResponse,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import (  # noqa: E501
                TextGenerationAsyncServiceStub,
            )
        from yandex.cloud.operation.operation_service_pb2 import GetOperationRequest
        from yandex.cloud.operation.operation_service_pb2_grpc import (
            OperationServiceStub,
        )
    except ImportError as e:
        raise ImportError(
            "Please install YandexCloud SDK  with `pip install yandexcloud` \
            or upgrade it to recent version."
        ) from e
    operation_api_url = "operation.api.cloud.yandex.net:443"
    channel_credentials = grpc.ssl_channel_credentials()
    async with grpc.aio.secure_channel(self.url, channel_credentials) as channel:
        request = CompletionRequest(
            model_uri=self.model_uri,
            completion_options=CompletionOptions(
                temperature=DoubleValue(value=self.temperature),
                max_tokens=Int64Value(value=self.max_tokens),
            ),
            messages=[Message(role="user", text=prompt)],
        )
        stub = TextGenerationAsyncServiceStub(channel)
        operation = await stub.Completion(request, metadata=self.grpc_metadata)  # type: ignore[attr-defined]
        async with grpc.aio.secure_channel(
            operation_api_url, channel_credentials
        ) as operation_channel:
            operation_stub = OperationServiceStub(operation_channel)
            while not operation.done:
                await asyncio.sleep(1)
                operation_request = GetOperationRequest(operation_id=operation.id)
                operation = await operation_stub.Get(
                    operation_request,
                    metadata=self.grpc_metadata,  # type: ignore[attr-defined]
                )

        completion_response = CompletionResponse()
        operation.response.Unpack(completion_response)
        return completion_response.alternatives[0].message.text


def _create_retry_decorator(llm: YandexGPT) -> Callable[[Any], Any]:
    from grpc import RpcError

    min_seconds = llm.sleep_interval
    max_seconds = 60
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type((RpcError))),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(llm: YandexGPT, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _completion_with_retry(**_kwargs: Any) -> Any:
        return _make_request(llm, **_kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(llm: YandexGPT, **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**_kwargs: Any) -> Any:
        return await _amake_request(llm, **_kwargs)

    return await _completion_with_retry(**kwargs)
