from langchain.agents.structured_output import ProviderStrategy


def test_provider_strategy_injects_title() -> None:
    """Test that ProviderStrategy injects a title into the inner schema if missing.

    This ensures compatibility with downstream tools (like ChatOpenAI.bind_tools)
    that require the schema to have a name/title when unwrapped.
    """
    # 1. Define a schema without a 'title'
    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    }

    # 2. Initialize ProviderStrategy
    strategy = ProviderStrategy(schema, strict=True)

    # 3. Get kwargs
    kwargs = strategy.to_model_kwargs()

    # 4. Verify structure
    assert "response_format" in kwargs
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"

    json_schema = response_format["json_schema"]
    assert "name" in json_schema
    assert "schema" in json_schema

    # 5. Verify title injection
    inner_schema = json_schema["schema"]
    assert "title" in inner_schema
    assert inner_schema["title"] == json_schema["name"]

    # 6. Verify strict mode is preserved
    assert json_schema["strict"] is True


def test_provider_strategy_preserves_existing_title() -> None:
    """Test that ProviderStrategy respects an existing title."""
    schema = {
        "title": "ExistingTitle",
        "type": "object",
        "properties": {"foo": {"type": "string"}},
    }

    strategy = ProviderStrategy(schema)
    kwargs = strategy.to_model_kwargs()

    inner_schema = kwargs["response_format"]["json_schema"]["schema"]
    assert inner_schema["title"] == "ExistingTitle"
