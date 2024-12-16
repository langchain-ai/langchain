"""Verify proxy imports from langchain to community are behaving as expected."""


def test_all_proxy_llms_are_llm_subclasses() -> None:
    """Simple test to make sure all things are subclasses of BaseLLM."""
    from langchain import llms
    from langchain_core.language_models import BaseLLM

    for cls in llms.__all__:
        assert issubclass(getattr(llms, cls), BaseLLM)


def test_vectorstores() -> None:
    """Simple test to make sure all things can be imported."""
    from langchain import vectorstores
    from langchain_core.vectorstores import VectorStore

    for cls in vectorstores.__all__:
        if cls not in [
            "AlibabaCloudOpenSearchSettings",
            "ClickhouseSettings",
            "MyScaleSettings",
            "AzureCosmosDBVectorSearch",
        ]:
            assert issubclass(getattr(vectorstores, cls), VectorStore)
