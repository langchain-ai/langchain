from langchain_upstage import GroundednessCheck


def test_langchain_upstage_groundedness_check() -> None:
    """Test Upstage Groundedness Check."""
    tool = GroundednessCheck()
    output = tool.run({"context": "foo bar", "query": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]


async def test_langchain_upstage_groundedness_check_async() -> None:
    """Test Upstage Groundedness Check asynchronous."""
    tool = GroundednessCheck()
    output = await tool.arun({"context": "foo bar", "query": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]
