from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config

InputOutputType = TypeVar("InputOutputType")


class BaseInjector(
    RunnableSerializable[InputOutputType, InputOutputType], Generic[InputOutputType]
):
    inject_objects: List[Any] = Field(init=False)
    """Objects which will be injected with an additional 
    attribute of 'attr_name'"""
    attr_name: str
    """Attribute name which will be injected to the objects"""
    pass_on_injection_fail: bool = False
    """Whether the injection will be skipped in the injection 
    failed or raise exceptions on injection failure"""

    def __init__(
            self, inject_objects: List[Any],
            attr_name: str,
            pass_on_injection_fail: bool = False
    ):
        for obj in inject_objects:
            if (
                    not hasattr(obj, attr_name)
                    and not pass_on_injection_fail
            ):
                raise ValueError(
                    f"Cannot inject '{attr_name}' "
                    f"to instances of class '{type(obj).__name__}'"
                    f" since the field {attr_name} is not defined. "
                    f"Either add field '{attr_name}' "
                    f"in class '{type(obj).__name__}'"
                    f" or set 'pass_on_injection_fail=True' "
                    f"as argument of the constructor of injector. "
                )
        super().__init__(inject_objects=inject_objects,
                         attr_name=attr_name,
                         pass_on_injection_fail=pass_on_injection_fail)

    def invoke(
            self, input: InputOutputType, config: Optional[RunnableConfig] = None
    ) -> InputOutputType:
        config = ensure_config(config)
        return self._call_with_config(
            self._pass_and_inject,
            input,
            config,
            run_type="inject",
        )

    def _pass_and_inject(self, input: InputOutputType) -> InputOutputType:
        """Return the output type of the injector."""
        for obj in self.inject_objects:
            if isinstance(obj, BaseModel):
                try:
                    setattr(obj, self.attr_name, input)
                except ValueError as value_err:
                    if self.pass_on_injection_fail:
                        pass
                    else:
                        raise ValueError(
                            f"Failed to inject attribute '{self.attr_name}' "
                            f"into instance of class '{obj.__class__.__name__}'. "
                            f"Either the class {obj.__class__.__name__} "
                            f"has no field '{self.attr_name}' defined"
                            f" or {obj.__class__.__name__} is immutable"
                        ) from value_err
                except TypeError as type_err:
                    if self.pass_on_injection_fail:
                        pass
                    else:
                        raise TypeError(
                            f"Failed to inject attribute '{self.attr_name}' "
                            f"into instance of class '{obj.__class__.__name__}'. "
                            f"Field '{self.attr_name}' "
                            f"of class {obj.__class__.__name__} is final. "
                        ) from type_err
            else:
                setattr(obj, self.attr_name, input)
        return input


class PromptInjector(BaseInjector[PromptValue]):
    """Inject PromptValue into given objects.
    After injection, the attribute 'prompt'
     of the given objects will be filled.
    If given object in 'inject_objects' is pydantic
    BaseModel and the attribute 'prompt' of the object
    is not defined, the injection will be failed on invoke and
    either an exception will be thrown (if pass_on_injection_fail is False),
    or the injection for this object
    will be skipped (if pass_on_injection_fail is True).

     Key init args:
         inject_objects: List[Any]
             List of the objects whose 'prompt' attribute will be injected.
         pass_on_injection_fail: bool
             Whether the injections of the given objects will be skipped,
             if the injections are failed. The injection will be failed once
             the given object does not have the 'prompt' attribute, or it is
             final. If the injection is not aloud to be skipped by defining
             the 'pass_on_injection_fail' as False, exceptions will be
             thrown on injection failures. Default value is False

     Examples:

         .. code-block:: python

             from langchain_core.prompts.injector import PromptInjector
             from some_random_package import RandomClassOne, RandomClassTwo
             objects_of_random_classes = [RandomClassOne(), RandomClassTwo()]

             injector = PromptInjector(
                 inject_objects = objects_of_random_classes
                 pass_on_injection_fail=True
             )

             prompt_value = ... # Create PromptValue object here
             injector.invoke(prompt)

             # Will throw AttributeError here, if 'prompt' was
             #not defined in RandomClassOne or RandomClassTwo,
             #since the injection is skipped on failure.
             print(objects_of_random_classes[0].prompt)
             print(objects_of_random_classes[1].prompt)

     Typically, the injectors are being used in chains to
     inject prompts into tools:

         .. code-block:: python

             from langchain_core.prompts.injector import PromptInjector

             prompt = ChatPromptTemplate.from_messages([
                 ...
             ])

             toolkit = ... # Get toolkit from langchain_community.agent_toolkits ...
             tools = toolkit.get_tools()
             llm_with_tools = ... # Bind tools to llm

             agent = (
                 | prompt
                 | PromptInjector(inject_objects=tools, pass_on_injection_fail=True)
                 | llm_with_tools
                 | OpenAIToolsAgentOutputParser()
             )
    """  # noqa: E501

    def __init__(self, inject_objects: List[Any], **kwargs: Any) -> None:
        super().__init__(inject_objects=inject_objects, attr_name="prompt", **kwargs)

    @property
    def InputType(self) -> Any:
        """Return the input type of the injector."""
        return Type[PromptValue]

    @property
    def OutputType(self) -> Any:
        """Return the output type of the injector."""
        return Type[PromptValue]
