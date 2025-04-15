"""Integration test for Sms."""

from langchain_community.utilities.twilio import TwilioAPIWrapper


def test_call() -> None:
    """Test that call runs."""
    twilio = TwilioAPIWrapper()
    output = twilio.run("Message", "+16162904619")
    assert output
