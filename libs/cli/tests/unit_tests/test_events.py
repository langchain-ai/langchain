import pytest

from langchain_cli.utils.events import EventDict, create_events


@pytest.mark.xfail(reason="Unknown reason")
def test_create_events() -> None:
    assert create_events(
        [EventDict(event="Test Event", properties={"test": "test"})]
    ) == {"status": "success"}
