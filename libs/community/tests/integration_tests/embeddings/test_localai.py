import pytest

from langchain_community.embeddings import LocalAIEmbeddings


@pytest.mark.requires("openai")
@pytest.mark.vcr
def test_localai_embed() -> None:
    llm = LocalAIEmbeddings(
        openai_api_key="foo", model="bge-m3", openai_api_base="https://foo.bar/v1"
    )
    eqq = llm.embed_query("foo bar")
    assert len(eqq) > 100
    assert eqq[0] != eqq[1]


@pytest.mark.requires("openai")
@pytest.mark.vcr
async def test_localai_aembed() -> None:
    llm = LocalAIEmbeddings(
        openai_api_key="foo", model="bge-m3", openai_api_base="https://foo.bar/v1"
    )
    eqq = await llm.aembed_query("foo bar")
    assert len(eqq) > 100
    assert eqq[0] != eqq[1]
