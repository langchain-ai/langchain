"""Integration test for Infobip Channels."""
from langchain_community.utilities.infobip import InfobipAPIWrapper


def test_sms() -> None:
    """Test that call runs."""
    infobip = InfobipAPIWrapper()
    output = infobip.run("Hello via Infobip", "+17706762438", "sms")
    assert output

def test_email() -> None:
    """Test that call runs."""
    infobip = InfobipAPIWrapper(from_email="voviro5448@giratex.com")
    output = infobip.run("Hello via Infobip", "voviro5448@giratex.com", "email")
    assert output