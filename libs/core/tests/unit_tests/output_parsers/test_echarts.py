from langchain_core.output_parsers.echarts import EChartsOutputParser


def test_echarts_output_parser() -> None:
    """Test ECharts output parser."""
    parser = EChartsOutputParser()
    text = """
    Here is the ECharts option for a simple bar chart:
    ```json
    {
      "title": {
        "text": "ECharts Example"
      },
      "tooltip": {},
      "xAxis": {
        "data": ["A", "B", "C"]
      },
      "yAxis": {},
      "series": [{
        "name": "sales",
        "type": "bar",
        "data": [5, 20, 36]
      }]
    }
    ```
    """
    output = parser.parse(text)
    assert isinstance(output, dict)
    assert output["title"]["text"] == "ECharts Example"
    assert output["xAxis"]["data"] == ["A", "B", "C"]
    assert output["series"][0]["name"] == "sales"


def test_echarts_output_parser_format_instructions() -> None:
    """Test ECharts output parser format instructions."""
    parser = EChartsOutputParser()
    instructions = parser.get_format_instructions()
    assert "Apache ECharts" in instructions
    assert "json" in instructions
    assert '"title": { "text": "ECharts Example" }' in instructions

if __name__=="__main__":
    test_echarts_output_parser()
