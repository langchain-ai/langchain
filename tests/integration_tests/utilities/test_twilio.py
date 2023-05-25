"""Integration test for Sms."""
from langchain.utilities.twilio import TwilioAPIWrapper


def test_call() -> None:
    """Test that call runs."""
    twilio = TwilioAPIWrapper()
    output = twilio.run("Message", "+16162904619")
    assert output
