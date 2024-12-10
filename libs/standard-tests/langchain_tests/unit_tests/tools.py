import os
from abc import abstractmethod
from typing import Tuple, Type, Union
from unittest import mock

import pytest
from langchain_core.tools import BaseTool
from pydantic import SecretStr

from langchain_tests.base import BaseStandardTests


class ToolsTests(BaseStandardTests):
    """
    :private:
    Base class for testing tools. This won't show in the documentation, but
    the docstrings will be inherited by subclasses.
    """

    @property
    @abstractmethod
    def tool_constructor(self) -> Union[Type[BaseTool], BaseTool]:
        """
        Returns a class or instance of a tool to be tested.
        """
        ...

    @property
    def tool_constructor_params(self) -> dict:
        """
        Returns a dictionary of parameters to pass to the tool constructor.
        """
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - it should not
        have {"name", "id", "args"} keys.
        """
        return {}

    @pytest.fixture
    def tool(self) -> BaseTool:
        """
        :private:
        """
        if isinstance(self.tool_constructor, BaseTool):
            if self.tool_constructor_params != {}:
                msg = (
                    "If tool_constructor is an instance of BaseTool, "
                    "tool_constructor_params must be empty"
                )
                raise ValueError(msg)
            return self.tool_constructor
        return self.tool_constructor(**self.tool_constructor_params)


class ToolsUnitTests(ToolsTests):
    """
    Base class for tools unit tests.
    """

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return {}, {}, {}

    def test_init(self) -> None:
        """
        Test that the tool can be initialized with :attr:`tool_constructor` and
        :attr:`tool_constructor_params`. If this fails, check that the
        keyword args defined in :attr:`tool_constructor_params` are valid.
        """
        if isinstance(self.tool_constructor, BaseTool):
            tool = self.tool_constructor
        else:
            tool = self.tool_constructor(**self.tool_constructor_params)
        assert tool is not None

    def test_init_from_env(self) -> None:
        env_params, tools_params, expected_attrs = self.init_from_env_params
        if env_params:
            with mock.patch.dict(os.environ, env_params):
                tool = self.tool_constructor(**tools_params)
            assert tool is not None
            for k, expected in expected_attrs.items():
                actual = getattr(tool, k)
                if isinstance(actual, SecretStr):
                    actual = actual.get_secret_value()
                assert actual == expected

    def test_has_name(self, tool: BaseTool) -> None:
        """
        Tests that the tool has a name attribute to pass to chat models.

        If this fails, add a `name` parameter to your tool.
        """
        assert tool.name

    def test_has_input_schema(self, tool: BaseTool) -> None:
        """
        Tests that the tool has an input schema.

        If this fails, add an `args_schema` to your tool.

        See
        `this guide <https://python.langchain.com/docs/how_to/custom_tools/#subclass-basetool>`_
        and see how `CalculatorInput` is configured in the
        `CustomCalculatorTool.args_schema` attribute
        """
        assert tool.get_input_schema()

    def test_input_schema_matches_invoke_params(self, tool: BaseTool) -> None:
        """
        Tests that the provided example params match the declared input schema.

        If this fails, update the `tool_invoke_params_example` attribute to match
        the input schema (`args_schema`) of the tool.
        """
        # this will be a pydantic object
        input_schema = tool.get_input_schema()

        assert input_schema(**self.tool_invoke_params_example)
