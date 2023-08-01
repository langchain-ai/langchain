"""Integration test for Segment."""
from langchain.utilities.segment import SegmentAPIWrapper


def test_call() -> None:
    """Test that call runs."""
    segment = SegmentAPIWrapper()
    segment.run(
        "Found Helpful",
        "user_id",
        dict(question_id="question_id", answer_id="answer_id", helpful=True),
    )
    assert True
