"""Test functionality for Pipeline tools."""
import json

import pytest

from langchain.tools import PipelineStep, PipelineTool


def test_step_in_str_out_str() -> None:
    f = PipelineStep(func=lambda x: x + "_processed")
    out = f("x")
    assert out == "x_processed"


def test_step_in_str_out_dict() -> None:
    f = PipelineStep(func=lambda x: x + "_processed", output_expression="y")
    out = f("x")
    assert out["y"] == "x_processed"


def test_step_in_dict_out_str() -> None:
    f = PipelineStep(func=lambda x: x + "_processed", input_expression={"x": "text"})
    out = f({"text": "hello"})
    assert out == "hello_processed"


def test_step_in_json_out_str() -> None:
    f = PipelineStep(func=lambda x: x + "_processed", input_expression={"x": "text"})
    out = f('{"text": "hello"}')
    assert out == "hello_processed"


def test_step_in_dict_out_dict() -> None:
    f = PipelineStep(
        func=lambda x: x + "_processed",
        input_expression={"x": "text"},
        output_expression="y",
    )
    out = f({"text": "hello"})
    assert out == {"text": "hello", "y": "hello_processed"}


def test_step_expression_with_in_str() -> None:
    with pytest.raises(ValueError):
        f = PipelineStep(
            func=lambda x: x + "_processed", input_expression={"x": "text"}
        )
        f("hello")


def test_step_output_expression() -> None:
    f = PipelineStep(
        func=lambda x: {"input": x, "output": x + "_p"},
        input_expression={"x": "text"},
        output_expression={"hello": "input", "world": "output"},
    )
    out = f({"text": "hello"})
    assert out == {"text": "hello", "hello": "hello", "world": "hello_p"}


def test_pipeline_run() -> None:
    f0 = PipelineStep(
        func=lambda x: x + 1,
        input_expression={"x": "num"},
        output_expression="o0",
    )
    f1 = PipelineStep(
        func=lambda x: x * 2, input_expression={"x": "o0"}, output_expression="o1"
    )
    f2 = PipelineStep(
        func=lambda x: x**2, input_expression={"x": "o0"}, output_expression="o2"
    )
    tool = PipelineTool(
        name="test-ppl", description="dummy for ppl test", steps=[f0, f1, f2]
    )

    out = tool('{"num": 2}')
    out = json.loads(out)
    assert out["o0"] == 3
    assert out["o1"] == 6
    assert out["o2"] == 9

    out = tool._run({"num": 2})
    out = json.loads(out)
    assert out["o0"] == 3
    assert out["o1"] == 6
    assert out["o2"] == 9


def test_pipeline_output_expression_list() -> None:
    f0 = PipelineStep(
        func=lambda x: x + 1,
        input_expression={"x": "num"},
        output_expression="o0",
    )
    f1 = PipelineStep(
        func=lambda x: x * 2, input_expression={"x": "o0"}, output_expression="o1"
    )
    f2 = PipelineStep(
        func=lambda x: x**2, input_expression={"x": "o0"}, output_expression="o2"
    )
    tool = PipelineTool(
        name="test-ppl",
        description="dummy for ppl test",
        steps=[f0, f1, f2],
        output_expression=["o1", "o2"],
    )

    out = tool('{"num": 2}')
    out = json.loads(out)
    assert out == {"o1": 6, "o2": 9}


def test_pipeline_output_single() -> None:
    f0 = PipelineStep(
        func=lambda x: x + 1,
        input_expression={"x": "num"},
        output_expression="o0",
    )
    f1 = PipelineStep(
        func=lambda x: x * 2, input_expression={"x": "o0"}, output_expression="o1"
    )
    f2 = PipelineStep(
        func=lambda x: x**2, input_expression={"x": "o0"}, output_expression="o2"
    )
    tool = PipelineTool(
        name="test-ppl",
        description="dummy for ppl test",
        steps=[f0, f1, f2],
        output_expression=["o2"],
    )

    out = tool('{"num": 2}')
    out = json.loads(out)
    assert out == 9
