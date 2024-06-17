from langchain_community.tools.custom_calculator import CustomCalculatorTool


def test_sum() -> None:
    tool = CustomCalculatorTool()
    result = tool.run({"a": 2, "b": 2})
    assert result == 4
