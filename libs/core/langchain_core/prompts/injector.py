from typing import Optional, Sequence, Any, List, Generic, TypeVar, Dict
from langchain_core.pydantic_v1 import root_validator
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableSerializable, RunnableConfig, ensure_config

InputOutputType = TypeVar("InputOutputType")


class BaseInjector(
    RunnableSerializable[InputOutputType, InputOutputType], Generic[InputOutputType]
):
    inject_objects: List[Any]
    attr_name: str = None

    @root_validator(pre=False, skip_on_failure=True)
    def validate_attr_name(cls, values: Dict) -> Dict:
        for obj in values["inject_objects"]:
            if hasattr(obj, values["attr_name"]):
                raise ValueError(
                    f"Cannot inject an already existing attribute named {values['attr_name']}. "
                    f" Please check if the attribute is already defined in class "
                    f"{type(obj).__name__} or if the object is being injected multiple times. "
                )
        return values

    @property
    def InputType(self) -> Any:
        """Return the input type of the injector."""
        return InputOutputType

    @property
    def OutputType(self) -> Any:
        """Return the output type of the injector."""
        return InputOutputType

    def invoke(
            self,
            input: PromptValue,
            config: Optional[RunnableConfig] = None
    ) -> PromptValue:
        config = ensure_config(config)
        if self.metadata:
            config["metadata"] = {**config["metadata"], **self.metadata}
        if self.tags:
            config["tags"] = config["tags"] + self.tags
        return self._call_with_config(
            lambda x: x,
            input,
            config,
            run_type="inject",
        )


class PromptInjector(BaseInjector[PromptValue]):
    def __init__(
        self,
        inject_objects: Sequence[Any],
    ) -> None:
        super().__init__(inject_objects=inject_objects)