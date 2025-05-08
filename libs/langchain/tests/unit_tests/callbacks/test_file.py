import pathlib
from typing import Any, Optional

import pytest

from langchain.callbacks import FileCallbackHandler
from langchain.chains.base import CallbackManagerForChainRun, Chain


class FakeChain(Chain):
    """Fake chain class for testing purposes."""

    be_correct: bool = True
    the_input_keys: list[str] = ["foo"]
    the_output_keys: list[str] = ["bar"]

    @property
    def input_keys(self) -> list[str]:
        """Input keys."""
        return self.the_input_keys

    @property
    def output_keys(self) -> list[str]:
        """Output key of bar."""
        return self.the_output_keys

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        return {"bar": "bar"}


def test_filecallback(capsys: pytest.CaptureFixture, tmp_path: pathlib.Path) -> Any:
    """Test the file callback handler."""
    p = tmp_path / "output.log"
    handler = FileCallbackHandler(str(p))
    chain_test = FakeChain(callbacks=[handler])
    chain_test.invoke({"foo": "bar"})
    # Assert the output is as expected
    assert p.read_text() == (
        "\n\n\x1b[1m> Entering new FakeChain "
        "chain...\x1b[0m\n\n\x1b[1m> Finished chain.\x1b[0m\n"
    )
