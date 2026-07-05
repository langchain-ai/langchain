from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel, __all__
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue

EXPECTED_ALL = [
    "BaseLanguageModel",
    "BaseChatModel",
    "SimpleChatModel",
    "BaseLLM",
    "LLM",
    "LangSmithParams",
    "LanguageModelInput",
    "LanguageModelOutput",
    "LanguageModelLike",
    "get_tokenizer",
    "LanguageModelLike",
    "FakeMessagesListChatModel",
    "FakeListChatModel",
    "GenericFakeChatModel",
    "FakeStreamingListLLM",
    "FakeListLLM",
    "ParrotFakeChatModel",
    "ModelProfile",
    "ModelProfileRegistry",
    "is_openai_data_block",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_pydantic_rebuild_handles_subclass_dict_method_shadowing_builtin() -> None:
    """Regression for Pydantic field collection with subclasses that define `dict()`.

    Pydantic 2.14.0a1 evaluates inherited field annotations during subclass
    rebuilds. If `BaseLanguageModel.metadata` uses a plain `dict[...]`
    annotation, the subclass `dict()` method can shadow the builtin and make
    annotation evaluation fail with `'function' object is not subscriptable`.
    """

    class DictMethodLanguageModel(BaseLanguageModel[str]):
        name: str = "test"

        def generate_prompt(
            self,
            prompts: list[PromptValue],
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
        ) -> LLMResult:
            raise NotImplementedError

        async def agenerate_prompt(
            self,
            prompts: list[PromptValue],
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
        ) -> LLMResult:
            raise NotImplementedError

        def dict(self, **_kwargs: Any) -> dict[str, Any]:
            return {}

    DictMethodLanguageModel.model_rebuild(force=True)

    assert "metadata" in DictMethodLanguageModel.model_fields
