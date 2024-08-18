from typing import Optional, Sequence, Any, List, Generic, TypeVar, Dict
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import root_validator
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableSerializable, RunnableConfig, ensure_config

InputOutputType = TypeVar("InputOutputType")


class BaseInjector(
    RunnableSerializable[InputOutputType, InputOutputType], Generic[InputOutputType]
):
    inject_objects: List[Any]
    attr_name: str = None
    pass_on_injection_fail: bool = False

    @root_validator(pre=False, skip_on_failure=True)
    def validate_attr_name(cls, values: Dict) -> Dict:
        for obj in values["inject_objects"]:
            if not hasattr(obj, values["attr_name"]) and not values["pass_on_injection_fail"]:
                raise ValueError(
                    f"Cannot inject '{values['attr_name']}' to instances of class '{type(obj).__name__}'"
                    f" since the field {values['attr_name']} is not defined. "
                    f"Either add field '{values['attr_name']}' in class '{type(obj).__name__}'"
                    f" or set 'pass_on_injection_fail=True' as argument of the constructor of injector. "
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
            input: InputOutputType,
            config: Optional[RunnableConfig] = None
    ) -> InputOutputType:
        config = ensure_config(config)
        return self._call_with_config(
            self._pass_and_inject,
            input,
            config,
            run_type="inject",
        )

    def _pass_and_inject(self, input: InputOutputType) -> InputOutputType:
        for obj in self.inject_objects:
            if isinstance(obj, BaseModel):
                try:
                    setattr(obj, self.attr_name, input)
                except ValueError:
                    if self.pass_on_injection_fail:
                        pass
                    else:
                        raise ValueError(f"Failed to inject attribute '{self.attr_name}' "
                                         f"into instance of class '{obj.__class__.__name__}'. "
                                         f"Either the class {obj.__class__.__name__} has no field '{self.attr_name}' defined"
                                         f" or {obj.__class__.__name__} is immutable")
                except TypeError:
                    if self.pass_on_injection_fail:
                        pass
                    else:
                        raise TypeError(f"Failed to inject attribute '{self.attr_name}' "
                                        f"into instance of class '{obj.__class__.__name__}'. "
                                        f"Field '{self.attr_name}' of class {obj.__class__.__name__} is final. ")
            else:
                setattr(obj, self.attr_name, input)
        return input


class PromptInjector(BaseInjector[PromptValue]):
    def __init__(
        self,
        inject_objects: Sequence[Any],
        **kwargs
    ) -> None:
        super().__init__(inject_objects=inject_objects, attr_name="prompt", **kwargs)
