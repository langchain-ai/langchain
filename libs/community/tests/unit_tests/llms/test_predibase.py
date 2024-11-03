from pydantic import SecretStr
from pytest import CaptureFixture

from langchain_community.llms.predibase import Predibase


def test_api_key_is_string() -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert isinstance(llm.predibase_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    print(llm.predibase_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_specifying_predibase_sdk_version_argument() -> None:
    llm = Predibase(  # type: ignore[call-arg]
        model="my_llm",
        predibase_api_key="secret-api-key",  # type: ignore[arg-type]
    )
    assert not llm.predibase_sdk_version

    legacy_predibase_sdk_version = "2024.4.8"
    llm = Predibase(  # type: ignore[call-arg]
        model="my_llm",
        predibase_api_key="secret-api-key",  # type: ignore[arg-type]
        predibase_sdk_version=legacy_predibase_sdk_version,
    )
    assert llm.predibase_sdk_version == legacy_predibase_sdk_version


def test_specifying_adapter_id_argument() -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert not llm.adapter_id

    llm = Predibase(  # type: ignore[call-arg]
        model="my_llm",
        predibase_api_key="secret-api-key",  # type: ignore[arg-type]
        adapter_id="my-hf-adapter",
    )
    assert llm.adapter_id == "my-hf-adapter"
    assert llm.adapter_version is None

    llm = Predibase(  # type: ignore[call-arg]
        model="my_llm",
        predibase_api_key="secret-api-key",  # type: ignore[arg-type]
        adapter_id="my-other-hf-adapter",
    )
    assert llm.adapter_id == "my-other-hf-adapter"
    assert llm.adapter_version is None


def test_specifying_adapter_id_and_adapter_version_arguments() -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert not llm.adapter_id

    llm = Predibase(  # type: ignore[call-arg]
        model="my_llm",
        predibase_api_key="secret-api-key",  # type: ignore[arg-type]
        adapter_id="my-hf-adapter",
        adapter_version=None,
    )
    assert llm.adapter_id == "my-hf-adapter"
    assert llm.adapter_version is None

    llm = Predibase(  # type: ignore[call-arg]
        model="my_llm",
        predibase_api_key="secret-api-key",  # type: ignore[arg-type]
        adapter_id="my-other-hf-adapter",
        adapter_version=3,
    )
    assert llm.adapter_id == "my-other-hf-adapter"
    assert llm.adapter_version == 3
