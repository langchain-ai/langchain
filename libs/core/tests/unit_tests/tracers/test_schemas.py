from langchain_core.tracers import schemas
from langchain_core.tracers.schemas import __all__ as schemas_all


def test_public_api() -> None:
    """Test for changes in the public API."""
    expected_all = [
        "Run",
    ]

    assert sorted(schemas_all) == expected_all

    # Assert that the object is actually present in the schema module
    for module_name in expected_all:
        assert hasattr(schemas, module_name)
        assert getattr(schemas, module_name) is not None
