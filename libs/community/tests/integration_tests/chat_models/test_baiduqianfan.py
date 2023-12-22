from typing import cast

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.chat_models.baidu_qianfan_endpoint import (
    QianfanChatEndpoint,
)


def test_qianfan_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("QIANFAN_AK", "test-api-key")
    monkeypatch.setenv("QIANFAN_SK", "test-secret-key")

    chat = QianfanChatEndpoint()
    print(chat.qianfan_ak, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.qianfan_sk, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_qianfan_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chat = QianfanChatEndpoint(
        qianfan_ak="test-api-key",
        qianfan_sk="test-secret-key",
    )
    print(chat.qianfan_ak, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.qianfan_sk, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secret_str() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chat = QianfanChatEndpoint(
        qianfan_ak="test-api-key",
        qianfan_sk="test-secret-key",
    )
    assert cast(SecretStr, chat.qianfan_ak).get_secret_value() == "test-api-key"
    assert cast(SecretStr, chat.qianfan_sk).get_secret_value() == "test-secret-key"
