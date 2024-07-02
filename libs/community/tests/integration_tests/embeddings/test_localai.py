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
def test_localai_embed_docs() -> None:
    llm = LocalAIEmbeddings(
        openai_api_key="foo", model="bge-m3", openai_api_base="https://foo.bar/v1"
    )
    docs1 = llm.embed_documents(["foo bar", "moo foo"])
    for doc in docs1:
        assert len(doc) > 100
        assert doc[0] != doc[1]
    assert len(docs1) == 2
    assert docs1[0] != docs1[1]
    docs2 = llm.embed_documents(["moo foo", "foo bar"])
    for doc in docs2:
        assert len(doc) > 100
        assert doc[0] != doc[1]
    assert len(docs2) == 2
    assert docs2[0] != docs2[1]

    assert docs1[0] == docs2[1]
    assert docs1[1] == docs2[0]


@pytest.mark.requires("openai")
@pytest.mark.vcr
async def test_localai_aembed() -> None:
    llm = LocalAIEmbeddings(
        openai_api_key="foo", model="bge-m3", openai_api_base="https://foo.bar/v1"
    )
    eqq = await llm.aembed_query("foo bar")
    assert len(eqq) > 100
    assert eqq[0] != eqq[1]


@pytest.mark.requires("openai")
@pytest.mark.vcr
async def test_localai_aembed_docs() -> None:
    llm = LocalAIEmbeddings(
        openai_api_key="foo", model="bge-m3", openai_api_base="https://foo.bar/v1"
    )
    docs1 = await llm.aembed_documents(["foo bar", "moo foo"])
    for doc in docs1:
        assert len(doc) > 100
        assert doc[0] != doc[1]
    assert len(docs1) == 2
    assert docs1[0] != docs1[1]
    docs2 = await llm.aembed_documents(["moo foo", "foo bar"])
    for doc in docs2:
        assert len(doc) > 100
        assert doc[0] != doc[1]
    assert len(docs2) == 2
    assert docs2[0] != docs2[1]

    assert docs1[0] == docs2[1]
    assert docs1[1] == docs2[0]
