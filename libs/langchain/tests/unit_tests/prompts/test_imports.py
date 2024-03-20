import pytest

from langchain import prompts
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "HumanMessagePromptTemplate",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "MessagesPlaceholder",
    "PipelinePromptTemplate",
    "Prompt",
    "PromptTemplate",
    "SemanticSimilarityExampleSelector",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
    "FewShotChatMessagePromptTemplate",
]
EXPECTED_DEPRECATED_IMPORTS = [
    "NGramOverlapExampleSelector",
]


def test_all_imports() -> None:
    assert set(prompts.__all__) == set(EXPECTED_ALL)
    assert_all_importable(prompts)


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(prompts, import_)
            assert "langchain_community" in e, f"{import_=} didn't error"
    with pytest.raises(AttributeError):
        getattr(prompts, "foo")
