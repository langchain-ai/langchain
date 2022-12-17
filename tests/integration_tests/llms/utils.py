"""Utils for LLM Tests."""

from langchain.llms.base import LLM


def assert_llm_equality(llm: LLM, loaded_llm: LLM) -> None:
    """Assert LLM Equality for tests."""
    # Check that they are the same type.
    assert type(llm) == type(loaded_llm)
    # Client field can be session based, so hash is different despite
    # all other values being the same, so just assess all other fields
    for field in llm.__fields__.keys():
        if field != "client" and field != "pipeline":
            val = getattr(llm, field)
            new_val = getattr(loaded_llm, field)
            assert new_val == val
