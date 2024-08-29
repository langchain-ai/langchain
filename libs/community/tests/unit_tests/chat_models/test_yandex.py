import os
from unittest import mock
from unittest.mock import MagicMock

import pytest

from langchain_community.chat_models.yandex import ChatYandexGPT


def test_yandexgpt_initialization() -> None:
    llm = ChatYandexGPT(
        iam_token="your_iam_token",  # type: ignore[arg-type]
        api_key="your_api_key",  # type: ignore[arg-type]
        folder_id="your_folder_id",
    )
    assert llm.model_name == "yandexgpt-lite"
    assert llm.model_uri.startswith("gpt://your_folder_id/yandexgpt-lite/")


def test_yandexgpt_model_params() -> None:
    llm = ChatYandexGPT(
        model_name="custom-model",
        model_version="v1",
        iam_token="your_iam_token",  # type: ignore[arg-type]
        api_key="your_api_key",  # type: ignore[arg-type]
        folder_id="your_folder_id",
    )
    assert llm.model_name == "custom-model"
    assert llm.model_version == "v1"
    assert llm.iam_token.get_secret_value() == "your_iam_token"
    assert llm.model_uri == "gpt://your_folder_id/custom-model/v1"


def test_yandexgpt_invalid_model_params() -> None:
    with pytest.raises(ValueError):
        ChatYandexGPT(model_uri="", iam_token="your_iam_token")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChatYandexGPT(
            iam_token="",  # type: ignore[arg-type]
            api_key="your_api_key",  # type: ignore[arg-type]
            model_uri="",
        )


@pytest.mark.parametrize(
    "api_key_or_token", [dict(api_key="bogus"), dict(iam_token="bogus")]
)
@pytest.mark.parametrize(
    "disable_logging",
    [dict(), dict(disable_request_logging=True), dict(disable_request_logging=False)],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_completion_call(api_key_or_token: dict, disable_logging: dict) -> None:
    absent_yandex_module_stub = MagicMock()
    grpc_mock = MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {
            "yandex.cloud.ai.foundation_models.v1."
            "text_common_pb2": absent_yandex_module_stub,
            "yandex.cloud.ai.foundation_models.v1.text_generation."
            "text_generation_service_pb2": absent_yandex_module_stub,
            "yandex.cloud.ai.foundation_models.v1.text_generation."
            "text_generation_service_pb2_grpc": absent_yandex_module_stub,
            "grpc": grpc_mock,
        },
    ):
        grpc_mock.RpcError = Exception
        stub = absent_yandex_module_stub.TextGenerationServiceStub
        request_stub = absent_yandex_module_stub.CompletionRequest
        msg_constructor_stub = absent_yandex_module_stub.Message
        args = {"folder_id": "fldr", **api_key_or_token, **disable_logging}
        ygpt = ChatYandexGPT(**args)
        grpc_call_mock = stub.return_value.Completion
        msg_mock = mock.Mock()
        msg_mock.message.text = "cmpltn"
        res_mock = mock.Mock()
        res_mock.alternatives = [msg_mock]
        grpc_call_mock.return_value = [res_mock]
        act_emb = ygpt.invoke("nomatter")
        assert act_emb.content == "cmpltn"
        assert len(grpc_call_mock.call_args_list) == 1
        once_called_args = grpc_call_mock.call_args_list[0]
        act_model_uri = request_stub.call_args_list[0].kwargs["model_uri"]
        act_text = msg_constructor_stub.call_args_list[0].kwargs["text"]
        act_metadata = once_called_args.kwargs["metadata"]
        assert "fldr" in act_model_uri
        assert act_text == "nomatter"
        assert act_metadata
        assert len(act_metadata) > 0
        if disable_logging.get("disable_request_logging"):
            assert ("x-data-logging-enabled", "false") in act_metadata
