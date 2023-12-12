"""Integration test for Connery API Wrapper."""
from langchain.tools.connery import ConneryService


def test_list_actions() -> None:
    """Test for listing Connery Actions."""
    connery = ConneryService()
    output = connery._list_actions()
    assert output is not None
    assert len(output) > 0


def test_get_action() -> None:
    """Test for getting Connery Action."""
    connery = ConneryService()
    # This is the ID of the preinstalled action "Refresh plugin cache"
    output = connery._get_action("CAF979E6D2FF4C8B946EEBAFCB3BA475")
    assert output is not None
    assert output.id == "CAF979E6D2FF4C8B946EEBAFCB3BA475"


def test_run_action() -> None:
    """Test for running Connery Action."""
    connery = ConneryService()
    output = connery._run_action("CAF979E6D2FF4C8B946EEBAFCB3BA475")
    assert output is not None
