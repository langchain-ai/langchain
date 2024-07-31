import pytest
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_community.llms.yandex import YandexGPT, _BaseYandexGPT


def test_yandexgpt_initialization() -> None:
    llm = YandexGPT(
        iam_token="your_iam_token", api_key="your_api_key", folder_id="your_folder_id"
    )
    assert llm.model_name == "yandexgpt-lite"
    assert llm.model_uri.startswith("gpt://your_folder_id/yandexgpt-lite/")


def test_yandexgpt_model_params() -> None:
    llm = YandexGPT(
        model_name="custom-model",
        model_version="v1",
        iam_token="your_iam_token",
        api_key="your_api_key",
        folder_id="your_folder_id",
    )
    assert llm.model_name == "custom-model"
    assert llm.model_version == "v1"
    assert llm.iam_token.get_secret_value() == "your_iam_token"
    assert llm.model_uri == "gpt://your_folder_id/custom-model/v1"


def test_yandexgpt_invalid_model_params() -> None:
    with pytest.raises(ValueError):
        YandexGPT(model_uri="", iam_token="your_iam_token")

    with pytest.raises(ValueError):
        YandexGPT(iam_token="", api_key="your_api_key", model_uri="")
