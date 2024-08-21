import os
from unittest import mock

import pytest

from langchain_community.llms.yandex import YandexGPT


def test_yandexgpt_initialization() -> None:
    llm = YandexGPT(
        iam_token="your_iam_token",  # type: ignore[arg-type]
        api_key="your_api_key",  # type: ignore[arg-type]
        folder_id="your_folder_id",
    )
    assert llm.model_name == "yandexgpt-lite"
    assert llm.model_uri.startswith("gpt://your_folder_id/yandexgpt-lite/")


def test_yandexgpt_model_params() -> None:
    llm = YandexGPT(
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
        YandexGPT(model_uri="", iam_token="your_iam_token")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        YandexGPT(
            iam_token="",  # type: ignore[arg-type]
            api_key="your_api_key",  # type: ignore[arg-type]
            model_uri="",
        )


@pytest.mark.parametrize("api_key_or_token", [dict(api_key="bogus"), dict(iam_token="bogus")])
@pytest.mark.parametrize(
    "disable_logging",
    [dict(), dict(disable_request_logging=True), dict(disable_request_logging=False)],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_completion_call(api_key_or_token, disable_logging) -> None:
    with mock.patch(
        "yandex.cloud.ai.foundation_models.v1.text_generation.text_generation_service_pb2_grpc.TextGenerationServiceStub"
    ) as stub:
        args = {"folder_id":"fldr", **api_key_or_token, **disable_logging}
        ygpt = YandexGPT(**args)
        grpc_call_mock = stub.return_value.Completion
        msg_mock = mock.Mock()
        msg_mock.message.text = "cmpltn"
        res_mock = mock.Mock()
        res_mock.alternatives = [msg_mock]
        grpc_call_mock.return_value = [res_mock]
        act_emb = ygpt.invoke("nomatter")
        assert act_emb == "cmpltn"
        assert len(grpc_call_mock.call_args_list) == 1
        once_called_args = grpc_call_mock.call_args_list[0]
        assert "fldr" in once_called_args.args[0].model_uri
        assert once_called_args.args[0].messages[0].text == "nomatter"
        assert once_called_args.kwargs["metadata"]
        assert len(once_called_args.kwargs["metadata"]) > 0
        if "disable_request_logging" in disable_logging and disable_logging["disable_request_logging"]:
            assert ("x-data-logging-enabled", "false") in once_called_args.kwargs[
                "metadata"
            ]
