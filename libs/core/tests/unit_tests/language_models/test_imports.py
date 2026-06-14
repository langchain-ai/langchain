from langchain_core.language_models import __all__

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


def test_transformers_not_imported_at_module_level() -> None:
    """GPT2TokenizerFast should not be imported until get_tokenizer() is called.

    Importing langchain_core.language_models.base must not trigger a
    `transformers` import, since that package is large and optional.
    """
    import sys

    # Ensure the module is importable without side-importing transformers
    # by checking that 'transformers' is not in sys.modules after the import.
    # (If transformers happens to be installed and already imported by another
    # test, we just skip the assertion — we can only prove absence, not presence.)
    before = "transformers" in sys.modules
    import importlib

    import langchain_core.language_models.base  # noqa: F401

    importlib.reload(langchain_core.language_models.base)
    after = "transformers" in sys.modules

    # If transformers was not present before the reload, it must not be present after.
    if not before:
        assert not after, (
            "`transformers` was imported at module level in "
            "langchain_core.language_models.base — it should be lazy."
        )
