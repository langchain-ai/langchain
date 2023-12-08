"""Test functionality related to prompt utils."""
from typing import Any, Dict

from langchain_core.prompt_values import StringPromptValue

from langchain.prompts.database.converter_prompt_template import ConverterPromptTemplate


def test_values() -> None:
    """Basic functionality test."""

    def converter(in1: str, in2: str) -> Dict[str, Any]:
        return {
            "out1": f"<{in1}>",
            "out2": in2 + in2,
        }

    prompt_fstring = "out1={out1} out2={out2} another={another}"

    c_p_template = ConverterPromptTemplate(
        template=prompt_fstring,
        input_variables=["another"],
        converter=lambda args_dict: converter(**args_dict),
        converter_input_variables=["in1", "in2"],
        converter_output_variables=["out1", "out2"],
    )

    result = c_p_template.format(in1="A", in2="x", another="a")
    assert result == "out1=<A> out2=xx another=a"


def test_validate_template() -> None:
    """Test for suppressing template validation."""

    def converter(a: str) -> Dict[str, Any]:
        return {"b": a}

    prompt_fstring = "b={b} another={another}"

    _ = ConverterPromptTemplate(
        template=prompt_fstring,
        input_variables=["another", "extraneous"],
        converter=lambda args_dict: converter(**args_dict),
        converter_input_variables=["a"],
        converter_output_variables=["b"],
        validate_template=False,
    )


def test_partialing() -> None:
    """Partialing in various combinations."""

    def converter(in1: int, in2: str) -> Dict[str, Any]:
        return {
            "out1": f"<{in1}>",
            "out2": in2 + in2,
        }

    prompt_fstring = "out1={out1} out2={out2} another={another}"

    c_p_template = ConverterPromptTemplate(
        template=prompt_fstring,
        input_variables=["another"],
        converter=lambda args_dict: converter(**args_dict),
        converter_input_variables=["in1", "in2"],
        converter_output_variables=["out1", "out2"],
    )

    c_p_partial1 = c_p_template.partial(in1="A")
    result1 = c_p_partial1.format(in2="x", another="a")
    assert result1 == "out1=<A> out2=xx another=a"

    c_p_partial2 = c_p_template.partial(another="a")
    result2 = c_p_partial2.format(in1="A", in2="x")
    assert result2 == "out1=<A> out2=xx another=a"

    c_p_partial3 = c_p_template.partial(in2="x", another="a")
    result3 = c_p_partial3.format(in1="A")
    assert result3 == "out1=<A> out2=xx another=a"


def test_converter_prompt_template_as_runnable() -> None:
    """Runnable interface test"""

    def mock_db_reader(user_id: str) -> Dict[str, Any]:
        return {
            "user_name": user_id.replace("_", " ").title(),
            "short_name": user_id[:2].upper(),
        }

    prompt_fstring = "ADJ={adj} USER_NAME={user_name} SHORT_NAME={short_name}"

    c_p_template = ConverterPromptTemplate(
        template=prompt_fstring,
        input_variables=["adj"],
        converter=lambda args_dict: mock_db_reader(**args_dict),
        converter_input_variables=["user_id"],
        converter_output_variables=["user_name", "short_name"],
    )

    invoke_result = c_p_template.invoke({"user_id": "john_doe", "adj": "sassy"})
    assert invoke_result == StringPromptValue(
        text="ADJ=sassy USER_NAME=John Doe SHORT_NAME=JO"
    )
