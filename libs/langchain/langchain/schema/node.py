from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
)

from typing_extensions import Unpack

from langchain.callbacks.manager import Callbacks


class BaseConfig(TypedDict, total=False):
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    tags: List[str]

    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    Keys should be strings, values should be JSON-serializable.
    """
    metadata: Dict[str, Any]

    """
    Callbacks for this call and any sub-calls (eg. a Chain calling an LLM).
    Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.
    """
    callbacks: Callbacks


Input = TypeVar("Input")
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output")


class Runnable(Protocol[Input, Output]):
    def invoke(self, input: Input, **kwargs: Unpack[BaseConfig]) -> Output:
        ...

    def batch(
        self,
        inputs: list[Input],
        config: Optional[BaseConfig | list[BaseConfig]] = None,
    ) -> list[Output]:
        ...

    def stream(self, input: Input, **kwargs: Unpack[BaseConfig]) -> Iterator[Output]:
        ...

    async def ainvoke(self, input: Input, **kwargs: Unpack[BaseConfig]) -> Output:
        ...

    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[BaseConfig | list[BaseConfig]] = None,
    ) -> list[Output]:
        ...

    def astream(
        self, input: Input, **kwargs: Unpack[BaseConfig]
    ) -> AsyncIterator[Output]:
        ...


class Me(Runnable[str, int]):
    def invoke(
        self,
        input: str,
        hello: str = "world",
        **kwargs: Unpack[BaseConfig],
    ) -> int:
        return 1

    def batch(
        self, inputs: list[str], config: Optional[BaseConfig | list[BaseConfig]] = None
    ) -> list[int]:
        return [1]

    def stream(self, input: str, **kwargs: Unpack[BaseConfig]) -> Iterator[int]:
        yield 1

    async def ainvoke(self, input: str, **kwargs: Unpack[BaseConfig]) -> int:
        return 1

    async def abatch(
        self, inputs: list[str], config: Optional[BaseConfig | list[BaseConfig]] = None
    ) -> list[int]:
        return [1]

    async def astream(
        self, input: str, **kwargs: Unpack[BaseConfig]
    ) -> AsyncIterator[int]:
        yield 1


me = Me()

me.invoke("hello")
