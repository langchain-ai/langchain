"""Test volc engine maas LLM model."""

from typing import Generator

from langchain_core.outputs import LLMResult
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.llms.volcengine_maas import (
    VolcEngineMaasBase,
    VolcEngineMaasLLM,
)


def test_api_key_is_string() -> None:
    llm = VolcEngineMaasBase(  # type: ignore[call-arg]
        volc_engine_maas_ak="secret-volc-ak",  # type: ignore[arg-type]
        volc_engine_maas_sk="secret-volc-sk",  # type: ignore[arg-type]
    )
    assert isinstance(llm.volc_engine_maas_ak, SecretStr)
    assert isinstance(llm.volc_engine_maas_sk, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = VolcEngineMaasBase(  # type: ignore[call-arg]
        volc_engine_maas_ak="secret-volc-ak",  # type: ignore[arg-type]
        volc_engine_maas_sk="secret-volc-sk",  # type: ignore[arg-type]
    )
    print(llm.volc_engine_maas_ak, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_default_call() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM()  # type: ignore[call-arg]
    output = llm.invoke("tell me a joke")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM()  # type: ignore[call-arg]
    output = llm.generate(["tell me a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_generate_stream() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM(streaming=True)  # type: ignore[call-arg]
    output = llm.stream("tell me a joke")
    assert isinstance(output, Generator)
