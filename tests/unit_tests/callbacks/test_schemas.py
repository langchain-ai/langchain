import langchain.callbacks.tracers.schemas as schemas
from langchain.callbacks.tracers.schemas import __all__ as schemas_all


def test_public_api() -> None:
    """Test for changes in the public API."""
    expected_all = [
        "BaseRun",
        "ChainRun",
        "LLMRun",
        "Run",
        "RunTypeEnum",
        "ToolRun",
        "TracerSession",
        "TracerSessionBase",
        "TracerSessionV1",
        "TracerSessionV1Base",
        "TracerSessionV1Create",
    ]

    assert sorted(schemas_all) == expected_all

    # Assert that the object is actually present in the schema module
    for module_name in expected_all:
        assert (
            hasattr(schemas, module_name) and getattr(schemas, module_name) is not None
        )
