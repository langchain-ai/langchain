from langchain_core.utils.tool_choice import normalize_tool_choice


def test_normalize_any_string():
    assert normalize_tool_choice("any") == "required"


def test_normalize_true_boolean():
    assert normalize_tool_choice(True) == "required"


def test_normalize_false_boolean():
    assert normalize_tool_choice(False) is None


def test_normalize_none():
    assert normalize_tool_choice(None) is None


def test_preserve_auto_and_required():
    assert normalize_tool_choice("auto") == "auto"
    assert normalize_tool_choice("required") == "required"
