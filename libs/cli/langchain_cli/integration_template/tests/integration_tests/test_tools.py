from typing import Type

from __module_name__.tools import __ModuleName__Tool
from langchain_tests.integration_tests import ToolsIntegrationTests


class TestParrotMultiplyToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[__ModuleName__Tool]:
        return __ModuleName__Tool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2, "b": 3}
