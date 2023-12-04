"""Utils for LLM Tests."""
from typing import List

from langchain.llms.base import BaseLLM


def assert_llm_equality(
    llm: BaseLLM, loaded_llm: BaseLLM, exclude: List[str] = []
) -> None:
    """Assert LLM Equality for tests."""
    # Check that they are the same type.
    assert type(llm) == type(loaded_llm)
    # Client field can be session based, so hash is different despite
    # all other values being the same, so just assess all other fields
    # Exclude contains fields name which can be session based.
    for field in llm.__fields__.keys():
        if field != "client" and field != "pipeline" and field not in exclude:
            val = getattr(llm, field)
            new_val = getattr(loaded_llm, field)
            assert new_val == val
