import pytest

from langchain.llms.bedrock import ALTERNATION_ERROR, _human_assistant_format

TEST_CASES = {
    """Hey""": """

Human: Hey

Assistant:""",
    """

Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """
Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """

Human: Human: Hello

Assistant:""": (
        "Error: Prompt must alternate between '\n\nHuman:' and '\n\nAssistant:'."
    ),
    """Human: Hello

Assistant: Hello

Human: Hello

Assistant:""": """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""",
    """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""": """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""",
    """

Human: Hello

Assistant: Hello

Human: Hello

Assistant: Hello

Assistant: Hello""": ALTERNATION_ERROR,
    """

Human: Hi.

Assistant: Hi.

Human: Hi.

Human: Hi.

Assistant:""": ALTERNATION_ERROR,
    """
Human: Hello""": """

Human: Hello

Assistant:""",
    """

Human: Hello
Hello

Assistant""": """

Human: Hello
Hello

Assistant

Assistant:""",
    """Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """Hello

Human: Hello

""": """Hello

Human: Hello



Assistant:""",
    """

Human: Assistant: Hello""": """

Human: 

Assistant: Hello""",
    """

Human: Human

Assistant: Assistant

Human: Assistant

Assistant: Human""": """

Human: Human

Assistant: Assistant

Human: Assistant

Assistant: Human""",
    """
Assistant: Hello there, your name is:

Human.

Human: Hello there, your name is: 

Assistant.""": """

Human: 

Assistant: Hello there, your name is:

Human.

Human: Hello there, your name is: 

Assistant.

Assistant:""",
    """

Human: Human: Hi

Assistant: Hi""": ALTERNATION_ERROR,
    """Human: Hi

Human: Hi""": ALTERNATION_ERROR,
    """

Assistant: Hi

Human: Hi""": """

Human: 

Assistant: Hi

Human: Hi

Assistant:""",
    """

Human: Hi

Assistant: Yo

Human: Hey

Assistant: Sup

Human: Hi

Assistant: Hi
Human: Hi
Assistant:""": """

Human: Hi

Assistant: Yo

Human: Hey

Assistant: Sup

Human: Hi

Assistant: Hi

Human: Hi

Assistant:""",
    """

Hello.

Human: Hello.

Assistant:""": """

Hello.

Human: Hello.

Assistant:""",
}


def test__human_assistant_format() -> None:
    for input_text, expected_output in TEST_CASES.items():
        if expected_output == ALTERNATION_ERROR:
            with pytest.warns(UserWarning, match=ALTERNATION_ERROR):
                _human_assistant_format(input_text)
        else:
            output = _human_assistant_format(input_text)
            assert output == expected_output
