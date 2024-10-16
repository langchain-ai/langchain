import langchain_core.errors as errors
import pytest


@pytest.fixture
def error_classes() -> list[type[errors.LangChainException]]:
    return list(errors.LangChainException.__subclasses__())
