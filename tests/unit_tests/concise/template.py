from langchain.concise.gemplate import template


def test_template():
    command = template("Please {{action}} the {{object}}.")
    # Test that the template is working.
    assert command(action="open", object="door") == "Please open the door."
    assert command(action="close") == "Please close the door."
    assert command(object="window") == "Please close the window."
    assert command() == "Please close the window."
