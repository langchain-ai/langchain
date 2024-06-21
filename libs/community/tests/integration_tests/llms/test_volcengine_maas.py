"""Test volc engine maas LLM model."""

from typing import Generator

from langchain_core.outputs import LLMResult
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture
from langchain_core.language_models.llms import Generation, LLMResult

from langchain_community.llms.volcengine_maas import (
    VolcEngineMaasBase,
    VolcEngineMaasLLM,
    VolcEngineMaasBaseV3,
    VolcEngineMaasLLMV3,
)


def test_api_key_is_string() -> None:
    llm = VolcEngineMaasBase(  # type: ignore[call-arg]
        volc_engine_maas_ak="secret-volc-ak",  # type: ignore[arg-type]
        volc_engine_maas_sk="secret-volc-sk",  # type: ignore[arg-type]
    )
    assert isinstance(llm.volc_engine_maas_ak, SecretStr)
    assert isinstance(llm.volc_engine_maas_sk, SecretStr)


def test_api_key_is_string_v3() -> None:
    llm = VolcEngineMaasBaseV3(  # type: ignore[call-arg]
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


def test_api_key_masked_when_passed_via_constructor_v3(
    capsys: CaptureFixture,
) -> None:
    llm = VolcEngineMaasBaseV3(  # type: ignore[call-arg]
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


def test_default_call_v3() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLMV3()  # type: ignore[call-arg]
    output = llm.invoke(
        [
            {"role": "user", "content": "tell me a joke"},
        ])
    assert len(output) == 1
    assert isinstance(output[0], str)

    output = llm.invoke(
        [
            {"role": "system", "content": "You are Doubao, an AI assistant developed by ByteDance."},
            {"role": "user", "content": "tell me a joke"},
        ])
    assert len(output) == 1
    assert isinstance(output[0], str)


def test_generate() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM()  # type: ignore[call-arg]
    output = llm.generate(["tell me a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_generate_v3() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLMV3()  # type: ignore[call-arg]
    output = llm.generate([{"role": "user", "content": "tell me a joke"}])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 1
    assert isinstance(output.generations[0], list)
    assert len(output.generations[0]) > 0
    assert isinstance(output.generations[0][0], Generation)


def test_generate_stream() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM(streaming=True)  # type: ignore[call-arg]
    output = llm.stream("tell me a joke")
    assert isinstance(output, Generator)


def test_generate_stream_v3() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLMV3(stream=True)  # type: ignore[call-arg]
    output = llm.stream([{"role": "user", "content": "tell me a joke"}])
    assert isinstance(output, Generator)
    result = ''.join(output)
    assert isinstance(result, str)
    assert len(result) > 0
