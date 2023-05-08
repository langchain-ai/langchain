import inspect
from langchain.llms.fake import FakeListLLM
from langchain.output_parsers.stitched import StitchedOutputParser


def test_stitched_output_parser_parse() -> None:
    stitch_chars = 50
    prompt = "Please provide an implementation for test_stitched_output_parser_parse:"
    full_response = f"Sure, here the implementation for test_stitched_output_parser_parse:\n\n```python\n{inspect.getsource(test_stitched_output_parser_parse)}```"
    partial_response_1 = full_response[: len(full_response) // 2]
    partial_response_2 = (
        f"Sorry about that! Here is a continuation of the response.\n\n"
        f"```python\n{partial_response_1[len(full_response) // 2:]}```"
    )
    partial_response_1_delete_chars = len("")
    partial_response_2_delete_chars = len(
        f"Sorry about that! Here is a continuation of the response.\n\n```python\n"
    )
    stitch = (
        partial_response_1[-stitch_chars:-partial_response_1_delete_chars]
        + partial_response_2[partial_response_2_delete_chars:stitch_chars]
    )
    llm_responses = [partial_response_2, stitch_chars]
    completion_validator_responses = [False, False, True]

    parser = StitchedOutputParser.from_llm(
        completion_validator=lambda: completion_validator_responses.pop(0),
        llm=FakeListLLM(responses=llm_responses),
        stitch_chars=stitch_chars,
    )

    # Test valid input
    response = parser.parse_with_prompt(completion=partial_response_1, prompt=prompt)
    assert response == full_response
