from typing import Any

import pytest
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)


class MyRunnable(RunnableSerializable[str, str]):
    my_property: str = Field(alias="my_property_alias")
    _my_hidden_property: str = ""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def my_error(cls, values: dict[str, Any]) -> Any:
        if "_my_hidden_property" in values:
            msg = "Cannot set _my_hidden_property"
            raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def build_extra(self) -> Self:
        self._my_hidden_property = self.my_property
        return self

    @override
    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return input + self._my_hidden_property

    def my_custom_function(self) -> str:
        return self.my_property

    def my_custom_function_w_config(
        self,
        config: RunnableConfig | None = None,
    ) -> str:
        _ = config
        return self.my_property

    def my_custom_function_w_kw_config(
        self,
        *,
        config: RunnableConfig | None = None,
    ) -> str:
        _ = config
        return self.my_property


class MyOtherRunnable(RunnableSerializable[str, str]):
    my_other_property: str

    @override
    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return input + self.my_other_property

    def my_other_custom_function(self) -> str:
        return self.my_other_property

    def my_other_custom_function_w_config(self, config: RunnableConfig) -> str:
        _ = config
        return self.my_other_property


def test_doubly_set_configurable() -> None:
    """Test that setting a configurable field with a default value works."""
    runnable = MyRunnable(my_property="a")
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="The property to test",
        )
    )

    assert configurable_runnable.invoke("d", config={"my_property": "c"}) == "dc"  # type: ignore[arg-type]


def test_alias_set_configurable() -> None:
    runnable = MyRunnable(my_property="a")
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property_alias",
            name="My property alias",
            description="The property to test alias",
        )
    )

    assert (
        configurable_runnable.invoke(
            "d", config=RunnableConfig(configurable={"my_property_alias": "c"})
        )
        == "dc"
    )


def test_field_alias_set_configurable() -> None:
    runnable = MyRunnable(my_property_alias="a")  # type: ignore[call-arg]
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property alias",
            description="The property to test alias",
        )
    )

    assert (
        configurable_runnable.invoke(
            "d", config=RunnableConfig(configurable={"my_property": "c"})
        )
        == "dc"
    )


def test_config_passthrough() -> None:
    runnable = MyRunnable(my_property="a")
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
    runnable = MyRunnable(my_property="a")
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
