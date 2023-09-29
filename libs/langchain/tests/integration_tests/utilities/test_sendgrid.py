"""Integration test for Email."""
from langchain.utilities.sendgrid import SendgridAPIWrapper


def test_call() -> None:
    """Test that call runs."""
    sendgrid = SendgridAPIWrapper()
    # from address must be from a verified sender to work. This is setup via sendgrid.com
    output = sendgrid.run(
        "test@test.com",
        "test@test.com",
        "langchain - test",
        " langchain FTW",
        "text/plain",
    )
    assert output == 202
