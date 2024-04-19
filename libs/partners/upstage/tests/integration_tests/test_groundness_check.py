from langchain_upstage import GroundednessCheck


def test_langchain_upstage_groundness_check() -> None:
    """Test Upstage Groundedness Check."""
    tool = GroundednessCheck()
    output = tool.run({"context": "foo bar", "assistant_message": "bar foo"})

    assert output.response_metadata["model_name"] == tool.api_wrapper.model_name


async def test_langchain_upstage_groundness_check_async() -> None:
    """Test Upstage Groundedness Check asynchronous."""
    tool = GroundednessCheck()
    output = await tool.arun({"context": "foo bar", "assistant_message": "bar foo"})

    assert output.response_metadata["model_name"] == tool.api_wrapper.model_name
