def test_import() -> None:
    """Test that the code can be imported"""
    from langchain.mcp import (  # noqa: F401, PLC0415
        callbacks,
        client,
        prompts,
        resources,
        tools,
    )
