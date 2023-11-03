import logging
import sys

import pytest

from langchain.callbacks.logging import LoggingCallbackHandler


def test_logging(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = logging.getLogger("test_logging")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    handler = LoggingCallbackHandler(logger)
    handler.on_text("test", color="red")

    # Assert logging actually took place
    assert len(caplog.record_tuples) == 1
    assert caplog.record_tuples[0][0] == logger.name
    assert caplog.record_tuples[0][1] == logging.INFO
    assert caplog.record_tuples[0][2] == "\x1b[31;1m\x1b[1;3mtest\x1b[0m"

    # Assert log handlers worked
    cap_result = capsys.readouterr()
    assert cap_result.out == "\x1b[31;1m\x1b[1;3mtest\x1b[0m\n"
