from langchain_core.errors import LangChainException, ErrorCode
import pytest


@pytest.fixture
def error_classes() -> list[type[LangChainException]]:
    return list(LangChainException.__subclasses__())


def test_error_classes_codes_one_one_mapping(
    error_classes: list[type[LangChainException]],
) -> None:
    all_error_codes_from_classes = {
        error_class("message").error_code for error_class in error_classes  # type: ignore
    }
    all_error_codes_from_enum = set(ErrorCode)

    assert all_error_codes_from_classes == all_error_codes_from_enum
