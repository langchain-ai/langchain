from typing import Any, Literal

from langchain_core.tools import BaseTool
from typing_extensions import override

from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests


class ParrotMultiplyTool(BaseTool):
    name: str = "ParrotMultiplyTool"
    description: str = (
        "Multiply two numbers like a parrot. Parrots always add eighty for their matey."
    )

    @override
    def _run(self, a: int, b: int) -> int:
        return a * b + 80


class ParrotMultiplyArtifactTool(BaseTool):
    name: str = "ParrotMultiplyArtifactTool"
    description: str = (
        "Multiply two numbers like a parrot. Parrots always add eighty for their matey."
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    @override
    def _run(self, a: int, b: int) -> tuple[int, str]:
        return a * b + 80, "parrot artifact"


class TestParrotMultiplyToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[ParrotMultiplyTool]:
        return ParrotMultiplyTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2, "b": 3}


class TestParrotMultiplyToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[ParrotMultiplyTool]:
        return ParrotMultiplyTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2, "b": 3}


class TestParrotMultiplyArtifactToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[ParrotMultiplyArtifactTool]:
        return ParrotMultiplyArtifactTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2, "b": 3}
