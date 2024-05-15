"""Test NLPCloud API wrapper."""

from pathlib import Path
from typing import cast

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.llms.loading import load_llm
from langchain_community.llms.nlpcloud import NLPCloud
from tests.integration_tests.llms.utils import assert_llm_equality


def test_nlpcloud_call() -> None:
    """Test valid call to nlpcloud."""
    llm = NLPCloud(max_length=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an NLPCloud LLM."""
    llm = NLPCloud(max_length=10)  # type: ignore[call-arg]
    llm.save(file_path=tmp_path / "nlpcloud.yaml")
    loaded_llm = load_llm(tmp_path / "nlpcloud.yaml")
    assert_llm_equality(llm, loaded_llm)


def test_nlpcloud_api_key(monkeypatch: MonkeyPatch, capsys: CaptureFixture) -> None:
    """Test that nlpcloud api key is a secret key."""
    # test initialization from init
    assert isinstance(NLPCloud(nlpcloud_api_key="1").nlpcloud_api_key, SecretStr)  # type: ignore[arg-type, call-arg]

    monkeypatch.setenv("NLPCLOUD_API_KEY", "secret-api-key")
    llm = NLPCloud()  # type: ignore[call-arg]
    assert isinstance(llm.nlpcloud_api_key, SecretStr)

    assert cast(SecretStr, llm.nlpcloud_api_key).get_secret_value() == "secret-api-key"

    print(llm.nlpcloud_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
