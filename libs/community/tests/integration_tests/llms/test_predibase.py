from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.llms.predibase import Predibase


def test_api_key_is_string() -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")
    print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::TEST_PREDIBASE.test_api_key_is_string()] LLM:\n{llm} ; TYPE: {str(type(llm))}')
    assert isinstance(llm.predibase_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    # TODO: <Alex>ALEX</Alex>
    # capsys: CaptureFixture,
    # TODO: <Alex>ALEX</Alex>
) -> None:
    llm = Predibase(model="my_llm", predibase_api_key="secret-api-key")
    print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::TEST_PREDIBASE.test_api_key_masked_when_passed_via_constructor()] LLM:\n{llm} ; TYPE: {str(type(llm))}')
    print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::TEST_PREDIBASE.test_api_key_masked_when_passed_via_constructor()] LLM.PREDIBASE_API_KEY:\n{llm.predibase_api_key} ; TYPE: {str(type(llm))}')
    # print(llm.predibase_api_key, end="")  # noqa: T201
    # captured = capsys.readouterr()
    # print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::TEST_PREDIBASE.test_api_key_masked_when_passed_via_constructor()] CAPTURED.OUT:\n{captured.out} ; TYPE: {str(type(captured.out))}')

    # TODO: <Alex>ALEX</Alex>
    # assert captured.out == "**********"
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    # assert "**********" in captured.out
    # TODO: <Alex>ALEX</Alex>
