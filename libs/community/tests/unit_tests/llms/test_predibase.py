from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.llms.predibase import Predibase


def test_api_key_is_string() -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")
    assert isinstance(llm.predibase_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")
    print(llm.predibase_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_specifying_adapter_id_argument() -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")
    assert not llm.adapter_id

    llm = Predibase(
        model="my_llm", predibase_api_key="secret-api-key", adapter_id="my-hf-adapter"
    )
    assert llm.adapter_id == "my-hf-adapter"

    llm = Predibase(
        model="my_llm",
        adapter_id="my-other-hf-adapter",
        predibase_api_key="secret-api-key",
    )
    assert llm.adapter_id == "my-other-hf-adapter"
