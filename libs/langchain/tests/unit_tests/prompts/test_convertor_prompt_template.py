"""Test functionality related to prompt utils."""
from typing import Any, Dict

from langchain_core.prompt_values import StringPromptValue

from langchain.prompts.database.convertor_prompt_template import ConvertorPromptTemplate


def test_values() -> None:
    """Basic functionality test."""

    def convertor(in1: int, in2: str) -> Dict[str, Any]:
        return {
            "out1": in1 * 2,
            "out2": in2 + in2,
        }

    prompt_fstring = "out1={out1} out2={out2} another={another}"

    c_p_template = ConvertorPromptTemplate(
        template=prompt_fstring,
        input_variables=["another"],
        convertor=lambda args_dict: convertor(**args_dict),
        convertor_input_variables=["in1", "in2"],
        convertor_output_variables=["out1", "out2"],
    )

    result = c_p_template.format(in1=5, in2="x", another="a")
    assert result == "out1=10 out2=xx another=a"


def test_validate_template() -> None:
    """Test for suppressing template validation."""

    def convertor(a: str) -> Dict[str, Any]:
        return {"b": a}

    prompt_fstring = "b={b} another={another}"

    _ = ConvertorPromptTemplate(
        template=prompt_fstring,
        input_variables=["another", "extraneous"],
        convertor=lambda args_dict: convertor(**args_dict),
        convertor_input_variables=["a"],
        convertor_output_variables=["b"],
        validate_template=False,
    )


def test_partialing() -> None:
    """Partialing in various combinations."""

    def convertor(in1: int, in2: str) -> Dict[str, Any]:
        return {
            "out1": in1 * 2,
            "out2": in2 + in2,
        }

    prompt_fstring = "out1={out1} out2={out2} another={another}"

    c_p_template = ConvertorPromptTemplate(
        template=prompt_fstring,
        input_variables=["another"],
        convertor=lambda args_dict: convertor(**args_dict),
        convertor_input_variables=["in1", "in2"],
        convertor_output_variables=["out1", "out2"],
    )

    c_p_partial1 = c_p_template.partial(in1=5)
    result1 = c_p_partial1.format(in2="x", another="a")
    assert result1 == "out1=10 out2=xx another=a"

    c_p_partial2 = c_p_template.partial(another="a")
    result2 = c_p_partial2.format(in1=5, in2="x")
    assert result2 == "out1=10 out2=xx another=a"

    c_p_partial3 = c_p_template.partial(in2="x", another="a")
    result3 = c_p_partial3.format(in1=5)
    assert result3 == "out1=10 out2=xx another=a"


def test_convertor_prompt_template_as_runnable() -> None:
    """Runnable interface test"""

    def mock_db_reader(user_id: str) -> Dict[str, Any]:
        return {
            "user_name": user_id.replace("_", " ").title(),
            "short_name": user_id[:2].upper(),
        }

    prompt_fstring = "ADJ={adj} USER_NAME={user_name} SHORT_NAME={short_name}"

    c_p_template = ConvertorPromptTemplate(
        template=prompt_fstring,
        input_variables=["adj"],
        convertor=lambda args_dict: mock_db_reader(**args_dict),
        convertor_input_variables=["user_id"],
        convertor_output_variables=["user_name", "short_name"],
    )

    invoke_result = c_p_template.invoke({"user_id": "john_doe", "adj": "sassy"})
    assert invoke_result == StringPromptValue(
        text="ADJ=sassy USER_NAME=John Doe SHORT_NAME=JO"
    )
