"""Integration test for Connery API Wrapper."""
from langchain_community.tools.connery import ConneryService


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


def test_run_action_with_no_iput() -> None:
    """Test for running Connery Action without input."""
    connery = ConneryService()
    # refreshPluginCache action from connery-io/connery-runner-administration plugin
    output = connery._run_action("CAF979E6D2FF4C8B946EEBAFCB3BA475")
    assert output is not None
    assert output == {}


def test_run_action_with_iput() -> None:
    """Test for running Connery Action with input."""
    connery = ConneryService()
    # summarizePublicWebpage action from connery-io/summarization-plugin plugin
    output = connery._run_action(
        "CA72DFB0AB4DF6C830B43E14B0782F70",
        {"publicWebpageUrl": "http://www.paulgraham.com/vb.html"},
    )
    assert output is not None
    assert output["summary"] is not None
    assert len(output["summary"]) > 0
