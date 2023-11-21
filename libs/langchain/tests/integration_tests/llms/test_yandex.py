import pytest
from pytest import MonkeyPatch

from langchain.llms.yandex import YandexGPT
from langchain.pydantic_v1 import SecretStr


@pytest.mark.requires("yandexcloud")
def test_yandex_gpt_constructor_with_secret(monkeypatch: MonkeyPatch) -> None:
    yandex_gpt = YandexGPT(iam_token="1")
    assert isinstance(yandex_gpt.iam_token, SecretStr)
    monkeypatch.setenv("YC_API_KEY", "yandex_api_key")
    yandex_gpt = YandexGPT(model_name="model_name", temperature=0.8, max_tokens=100)
    assert isinstance(yandex_gpt.api_key, SecretStr)


@pytest.mark.requires("yandexcloud")
def test_yandex_gpt_constructor_wo_secret() -> None:
    with pytest.raises(
        ValueError,
        match="Either 'YC_API_KEY' or 'YC_IAM_TOKEN' must be provided.",
    ):
        _ = YandexGPT()
