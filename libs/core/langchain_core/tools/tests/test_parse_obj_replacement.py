"""Test replacement of `parse_obj` with `model_validate` in BaseTool."""

from pydantic import BaseModel

from langchain_core.tools.base import BaseTool


class DummyArgs(BaseModel):
    """Dummy input model for testing."""

    name: str


class DummyTool(BaseTool):
    """Tool using DummyArgs as args_schema."""

    args_schema: type[DummyArgs] = DummyArgs

    def _run(self, name: str) -> str:
        """Simple echo tool."""
        return f"Hello {name}"


def test_model_validate() -> None:
    """Test if tool handles args_schema using model_validate correctly."""
    tool = DummyTool(name="dummy", description="dummy tool")
    output = tool.invoke({"name": "LangChain"})
    assert output == "Hello LangChain"  # noqa: S101
