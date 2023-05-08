from langchain.llms.fake import FakeListLLM
from langchain.output_parsers.stitched import StitchedOutputParser


def test_stitched_output_parser_parse() -> None:
    stitch_len = 10
    irrelevant_len = 2
    prompt = "Please provide an implementation for test_stitched_output_parser_parse:"
    split_pos = 7
    original_response = split_pos * "A" + 12 * "B"

    partial_response_1 = original_response[:split_pos] + irrelevant_len * "C"
    partial_response_2 = irrelevant_len * "D" + original_response[split_pos:]

    stitch_start = split_pos - stitch_len // 2 + irrelevant_len
    stitch_end = split_pos + stitch_len // 2 - irrelevant_len
    stitch = original_response[stitch_start:stitch_end]
    llm_responses = [partial_response_2, stitch]
    completion_validator_responses = [False, True]

    parser = StitchedOutputParser.from_llm(
        completion_validator=lambda _: completion_validator_responses.pop(0),
        llm=FakeListLLM(responses=llm_responses),
        stitch_chars=stitch_len,
    )

    # Test valid input
    stitched_response = parser.parse_with_prompt(
        completion=partial_response_1, prompt=prompt
    )
    # print(
    #     dedent(
    #         f"""
    #         --- test_stitched_output_parser_parse ---
    #         origonal_response:  {original_response}
    #         stitched_response:  {stitched_response}
    #         partial_response_1: {partial_response_1}
    #         partial_response_2: {(split_pos-irrelevant_len)*' '+partial_response_2}
    #         stitch:             {(split_pos-stitch_len//2)*' '+stitch}
    #         -----------------------------------------
    #         """
    #     )
    # )

    assert stitched_response == original_response
