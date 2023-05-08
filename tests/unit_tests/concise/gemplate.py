from langchain.concise.gemplate import gemplate
from langchain.llms.fake import FakeListLLM


def test_gemplate():
    outputs = [
        "We would always hang out after school...",
        "Once upon a time, two people...",
        "It was a dark and stormy night...",
    ]
    command = gemplate(
        "Tell me a {{story_type}} about {{object}}:", llm=FakeListLLM(outputs)
    )
    # Test that the gemplate is working.
    assert command(action="story", object="love") == outputs[0]
    assert command(action="tale") == outputs[1]
    assert command(object="tornados") == outputs[2]
