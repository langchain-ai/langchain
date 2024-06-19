from typing import Any, Optional

import pytest

from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)


class MyRunnable(RunnableSerializable[str, str]):
    my_property: str
    _my_hidden_property: str = ""

    def __post_init__(self) -> None:
        self._my_hidden_property = self.my_property

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> Any:
        return input + self._my_hidden_property

    def my_custom_function(self) -> str:
        return self.my_property

    def my_custom_function_w_config(
        self, config: Optional[RunnableConfig] = None
    ) -> str:
        return self.my_property

    def my_custom_function_w_kw_config(
        self, *, config: Optional[RunnableConfig] = None
    ) -> str:
        return self.my_property


class MyOtherRunnable(RunnableSerializable[str, str]):
    my_other_property: str

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> Any:
        return input + self.my_other_property

    def my_other_custom_function(self) -> str:
        return self.my_other_property

    def my_other_custom_function_w_config(self, config: RunnableConfig) -> str:
        return self.my_other_property


def test_doubly_set_configurable() -> None:
    """Test that setting a configurable field with a default value works"""
    runnable = MyRunnable(my_property="a")  # type: ignore
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="The property to test",
        )
    )

    assert (
        configurable_runnable.invoke(
            "d", config=RunnableConfig(configurable={"my_property": "c"})
        )
        == "dc"
    )


def test_config_passthrough() -> None:
    runnable = MyRunnable(my_property="a")  # type: ignore
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="The property to test",
        )
    )
    # first one
    with pytest.raises(AttributeError):
        configurable_runnable.not_my_custom_function()  # type: ignore[attr-defined]

    assert configurable_runnable.my_custom_function() == "a"  # type: ignore[attr-defined]
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            {"configurable": {"my_property": "b"}}
        )
        == "b"
    )
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "b"}}
        )
        == "b"
    )

    # second one
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function()  # type: ignore[attr-defined]
        == "b"
    )


def test_config_passthrough_nested() -> None:
    runnable = MyRunnable(my_property="a")  # type: ignore
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="The property to test",
        )
    ).configurable_alternatives(
        ConfigurableField(id="which", description="Which runnable to use"),
        other=MyOtherRunnable(my_other_property="c"),
    )
    # first one
    with pytest.raises(AttributeError):
        configurable_runnable.not_my_custom_function()  # type: ignore[attr-defined]
    assert configurable_runnable.my_custom_function() == "a"  # type: ignore[attr-defined]
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            {"configurable": {"my_property": "b"}}
        )
        == "b"
    )
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "b"}}
        )
        == "b"
    )
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function()  # type: ignore[attr-defined]
        == "b"
    ), "function without config can be called w bound config"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "func with config arg can be called w bound config without config"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "func with config arg can be called w bound config with config as kwarg"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_kw_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "function with config kwarg can be called w bound config w/out config"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_kw_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "function with config kwarg can be called w bound config with config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function()  # type: ignore[attr-defined]
        == "b"
    ), "function without config can be called w bound config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "func with config arg can be called w bound config without config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "func with config arg can be called w bound config with config as kwarg"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_kw_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "function with config kwarg can be called w bound config w/out config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_kw_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "function with config kwarg can be called w bound config with config"
    # second one
    with pytest.raises(AttributeError):
        configurable_runnable.my_other_custom_function()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        configurable_runnable.my_other_custom_function_w_config(  # type: ignore[attr-defined]
            {"configurable": {"my_other_property": "b"}}
        )
    with pytest.raises(AttributeError):
        configurable_runnable.with_config(
            configurable={"my_other_property": "c", "which": "other"}
        ).my_other_custom_function()  # type: ignore[attr-defined]
