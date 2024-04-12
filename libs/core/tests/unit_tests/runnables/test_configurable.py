from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)


class MyRunnable(RunnableSerializable[str, str]):
    my_property: str = Field(alias="my_property_alias")
    _my_hidden_property: str = ""

    class Config:
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def my_error(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "_my_hidden_property" in values:
            raise ValueError("Cannot set _my_hidden_property")
        return values

    @root_validator()
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["_my_hidden_property"] = values["my_property"]
        return values

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> Any:
        return input + self._my_hidden_property


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


def test_alias_set_configurable() -> None:
    runnable = MyRunnable(my_property="a")  # type: ignore
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
    runnable = MyRunnable(my_property_alias="a")
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
