import logging
import sys
import uuid

import pytest

from langchain.callbacks.tracers import LoggingCallbackHandler


def test_logging(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    # Set up a Logger and a handler so we can check the Logger's handlers work too
    logger = logging.getLogger("test_logging")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    handler = LoggingCallbackHandler(logger, extra={"test": "test_extra"})
    handler.on_text("test", run_id=uuid.uuid4())

    # Assert logging actually took place
    assert len(caplog.record_tuples) == 1
    record = caplog.records[0]
    assert record.name == logger.name
    assert record.levelno == logging.INFO
    assert (
        record.msg == "\x1b[36;1m\x1b[1;3m[text]\x1b[0m \x1b[1mNew text:\x1b[0m\ntest"
    )
    # Check the extra shows up
    assert record.test == "test_extra"  # type: ignore[attr-defined]

    # Assert log handlers worked
    cap_result = capsys.readouterr()
    assert (
        cap_result.out
        == "\x1b[36;1m\x1b[1;3m[text]\x1b[0m \x1b[1mNew text:\x1b[0m\ntest\n"
    )
