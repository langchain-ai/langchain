"""Integration test for Infobip Channels."""
from langchain_community.utilities.infobip import InfobipAPIWrapper


def test_sms() -> None:
    """Test that call runs."""
    infobip = InfobipAPIWrapper()
    output = infobip.run(message="Hello via Infobip", to="+17706762438")
    assert output

def test_email() -> None:
    """Test that call runs."""
    infobip = InfobipAPIWrapper()
    output = infobip.run(message="Hello via Infobip", to="voviro5448@giratex.com", subject="Test")
    assert output