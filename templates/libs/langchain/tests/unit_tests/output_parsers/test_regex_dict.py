"""Test in memory docstore."""
from langchain.output_parsers.regex_dict import RegexDictParser

DEF_EXPECTED_RESULT = {"action": "Search", "action_input": "How to use this class?"}

DEF_OUTPUT_KEY_TO_FORMAT = {"action": "Action", "action_input": "Action Input"}

DEF_README = """We have just received a new result from the LLM, and our next step is
to filter and read its format using regular expressions to identify specific fields,
such as:

- Action: Search
- Action Input: How to use this class?
- Additional Fields: "N/A"

To assist us in this task, we use the regex_dict class. This class allows us to send a
dictionary containing an output key and the expected format, which in turn enables us to
retrieve the result of the matching formats and extract specific information from it.

To exclude irrelevant information from our return dictionary, we can instruct the LLM to
use a specific command that notifies us when it doesn't know the answer. We call this
variable the "no_update_value", and for our current case, we set it to "N/A". Therefore,
we expect the result to only contain the following fields:
{
 {key = action, value = search}
 {key = action_input, value = "How to use this class?"}.
}"""


def test_regex_dict_result() -> None:
    """Test regex dict result."""
    regex_dict_parser = RegexDictParser(
        output_key_to_format=DEF_OUTPUT_KEY_TO_FORMAT, no_update_value="N/A"
    )
    result_dict = regex_dict_parser.parse(DEF_README)
    print("parse_result:", result_dict)
    assert DEF_EXPECTED_RESULT == result_dict
