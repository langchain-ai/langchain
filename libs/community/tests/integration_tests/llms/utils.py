"""Utils for LLM Tests."""

from langchain_core.language_models.llms import BaseLLM


def assert_llm_equality(llm: BaseLLM, loaded_llm: BaseLLM) -> None:
    """Assert LLM Equality for tests."""
    # Check that they are the same type.
    assert type(llm) is type(loaded_llm)
    # Client field can be session based, so hash is different despite
    # all other values being the same, so just assess all other fields
    for field in llm.__fields__.keys():
        if field != "client" and field != "pipeline":
            val = getattr(llm, field)
            new_val = getattr(loaded_llm, field)
            assert new_val == val
