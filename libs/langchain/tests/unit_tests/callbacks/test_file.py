import pathlib
from typing import Optional

from langchain_core.callbacks import CallbackManagerForChainRun

from langchain.callbacks import FileCallbackHandler
from langchain.chains.base import Chain


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


def test_filecallback(tmp_path: pathlib.Path) -> None:
    """Test the file callback handler."""
    log1 = tmp_path / "output.log"
    handler = FileCallbackHandler(str(log1))
    chain_test = FakeChain(callbacks=[handler])
    chain_test.invoke({"foo": "bar"})
    handler.close()
    # Assert the output is as expected
    assert "Entering new FakeChain chain" in log1.read_text()

    # Test using a callback manager
    log2 = tmp_path / "output2.log"

    with FileCallbackHandler(str(log2)) as handler_cm:
        chain_test = FakeChain(callbacks=[handler_cm])
        chain_test.invoke({"foo": "bar"})

    assert "Entering new FakeChain chain" in log2.read_text()

    # Test passing via invoke callbacks

    log3 = tmp_path / "output3.log"

    with FileCallbackHandler(str(log3)) as handler_cm:
        chain_test.invoke({"foo": "bar"}, {"callbacks": [handler_cm]})
    assert "Entering new FakeChain chain" in log3.read_text()
