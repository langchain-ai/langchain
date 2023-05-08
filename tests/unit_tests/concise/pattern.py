from langchain.concise.pattern import pattern
from langchain.llms.fake import FakeListLLM


def test_pattern():
    examples = [
        ("A pet or child is detected in the path", "stop and wait"),
        ("Brushes become tangled or stuck", "pause and alert the user"),
        ("The dustbin becomes full", "stop and return to the charging station"),
        ("A spill is detected", "pause and clean up spill"),
        ("High dust or debris is detected", "spend extra time cleaning"),
        ("The brushes become too dirty", "switch to a more powerful cleaning mode"),
        ("A new surface type is encountered", "pause and adjust cleaning mode"),
        ("A malfunction is detected", "pause and alert the user"),
        ("The battery is low", "pause and return to charging station"),
        ("An environment change is detected", "pause and adjust cleaning route"),
    ]

    result = pattern(
        "There is ice cream on the floor",
        examples=examples,
        llm=FakeListLLM(["Mop the ice cream"]),
    )
    assert isinstance(result, str)
    assert len(result) > 0
