from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from pydantic import Field

from langchain.load.serializable import Serializable
from langchain.schema.language_model import LanguageModelInput, LanguageModelOutput
from langchain.schema.runnable import RunnableConfig
from langchain.schema.runnable.base import Other, Input, RunnableSequence, Output

RunnableLike = Union[
    Runnable[Any, Other],
    Callable[[Any], Other],
    Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other]]],
]

class Runnable(Generic[Input, Output]):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        ...

    def batch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> List[Output]:
        ...

    def stream(
            self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        ...

    def __or__(self, other: RunnableLike) -> RunnableSequence[Input, Other]:
        ...

class Document(Serializable):
    page_content: str
    metadata: dict = Field(default_factory=dict)

class BaseLoader:
    def load(self) -> List[Document]:
        ...

    def lazy_load(self) -> Iterator[Document]:
        ...

class Embeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...

class VectorStore:
    def similarity_search(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        ...

    def similarity_search_with_score(
        self, query: str, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        ...

    def max_marginal_relevance_search(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        ...

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        ...

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        ...

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        ...

class BaseRetriever(Serializable, Runnable[str, List[Document]]):

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        return self.get_relevant_documents(input, **(config or {}))

    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        ...

class BaseMessage(Serializable):
    """The base abstract Message class.

    Messages are the inputs and outputs of ChatModels.
    """

    content: str
    """The string contents of the message."""

    additional_kwargs: dict = Field(default_factory=dict)
    """Any additional information."""

class BaseLanguageModel(
    Serializable, Runnable[LanguageModelInput, LanguageModelOutput]
):
    def generate_prompt(
        self, prompts: List[PromptValue], **kwargs: Any,
    ) -> LLMResult:
        ...

    def predict(self, text: str, **kwargs: Any) -> str:
        ...

    def predict_messages(
        self, messages: List[BaseMessage], **kwargs: Any,
    ) -> BaseMessage:
        ...


class BaseLLM(BaseLanguageModel[str]):
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        result = self.generate([prompt], **kwargs,)
        return result.generations[0][0].text

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig]=None,
        **kwargs: Any,
    ) -> str:
        _kwargs = {**(config or {}), **kwargs}
        result = self.generate_prompt([self._convert_input(input)], **_kwargs)
        return result.generations[0][0].text

    def batch(self, inputs: List[LanguageModelInput], **kwargs: Any) -> List[str]:
        ...

    def stream(self, input: LanguageModelInput, **kwargs: Any) -> Iterator[str]:
        ...

    def generate(self, prompts: List[str], **kwargs: Any,) -> LLMResult:
        ...

    def generate_prompt(self, prompts: List[PromptValue], **kwargs: Any,) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return self.generate(prompt_strings, **kwargs)

    def predict(self, text: str, **kwargs: Any) -> str:
        return self(text, **kwargs)

    def predict_messages(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> BaseMessage:
        text = get_buffer_string(messages)
        content = self(text, **kwargs)
        return AIMessage(content=content)

class BaseChatModel(BaseLanguageModel[BaseMessageChunk], ABC):

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseMessageChunk:
        _kwargs = {**(config or {}), **kwargs}
        result = self.generate_prompt([self._convert_input(input)], **_kwargs)
        return result.generations[0][0].message

    def batch(self, inputs: List[LanguageModelInput], **kwargs: Any) -> List[str]:
        ...

    def stream(
        self, input: LanguageModelInput, **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        ...

    def generate(
        self, messages: List[List[BaseMessage]], **kwargs: Any,
    ) -> LLMResult:
        ...

    def __call__(self, messages: List[BaseMessage], **kwargs: Any,) -> BaseMessage:
        return self.generate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        ).generations[0][0]

    def generate_prompt(self, prompts: List[PromptValue], **kwargs: Any) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, **kwargs)

    def predict(self, text: str,  **kwargs: Any) -> str:
        result = self([HumanMessage(content=text)] **kwargs)
        return result.content

    def predict_messages(
        self, messages: List[BaseMessage], **kwargs: Any,
    ) -> BaseMessage:
        return self(messages,  **kwargs)

class BaseMemory(Serializable):
    @property
    def memory_variables(self) -> List[str]:
        ...

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        ...

    def clear(self) -> None:
        ...

class BaseChatMessageHistory:
    messages: List[BaseMessage]

    def add_message(self, message: BaseMessage) -> None:
        ...

    def clear(self) -> None:
        ...
