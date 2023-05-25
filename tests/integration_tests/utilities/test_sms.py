"""Integration test for Sms."""
from langchain.utilities.sms import Sms


def test_call() -> None:
    """Test that call runs."""
    chain = Sms()
    output = chain.run("Message", "+16162904619")
    assert output
