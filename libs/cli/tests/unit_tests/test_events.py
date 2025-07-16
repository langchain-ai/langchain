import pytest

from langchain_cli.utils.events import EventDict, create_events


@pytest.mark.xfail(reason="Unknown reason")
def test_create_events() -> None:
    result = create_events([EventDict(event="Test Event", properties={"test": "test"})])
    if result != {"status": "success"}:
        msg = "Expected {'status': 'success'}, got " + repr(result)
        raise ValueError(msg)
