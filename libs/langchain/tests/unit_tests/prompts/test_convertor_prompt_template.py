"""Test functionality related to prompt utils."""
from typing import Any, Dict

from langchain.prompts.database.convertor_prompt_template import ConvertorPromptTemplate


def test_values() -> None:
    """Basic functionality test."""
    def convertor(in1: int, in2: str) -> Dict[str, Any]:
        return {
            'out1': in1 * 2,
            'out2': in2 + in2,
        }

    prompt_fstring = "out1={out1} out2={out2} another={another}"

    c_p_template = ConvertorPromptTemplate(
            template=prompt_fstring,
            input_variables=['another'],
            convertor=lambda args_dict: convertor(**args_dict),
            convertor_input_variables=['in1', 'in2'],
            convertor_output_variables=['out1', 'out2'],
        )

    result = c_p_template.format(in1=5, in2='x', another='a')
    assert result == "out1=10 out2=xx another=a"


def test_partialing() -> None:
    """Partialing in various combinations."""
    def convertor(in1: int, in2: str) -> Dict[str, Any]:
        return {
            'out1': in1 * 2,
            'out2': in2 + in2,
        }

    prompt_fstring = "out1={out1} out2={out2} another={another}"

    c_p_template = ConvertorPromptTemplate(
            template=prompt_fstring,
            input_variables=['another'],
            convertor=lambda args_dict: convertor(**args_dict),
            convertor_input_variables=['in1', 'in2'],
            convertor_output_variables=['out1', 'out2'],
        )

    c_p_partial1 = c_p_template.partial(in1=5)
    result1 = c_p_partial1.format(in2='x', another='a')
    assert result1 == "out1=10 out2=xx another=a"

    c_p_partial2 = c_p_template.partial(another='a')
    result2 = c_p_partial2.format(in1=5, in2='x')
    assert result2 == "out1=10 out2=xx another=a"

    c_p_partial3 = c_p_template.partial(in2='x', another='a')
    result3 = c_p_partial3.format(in1=5)
    assert result3 == "out1=10 out2=xx another=a"
