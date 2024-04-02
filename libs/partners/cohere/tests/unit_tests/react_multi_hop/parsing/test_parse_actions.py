from typing import Any, Dict, List, Optional

import pytest

from langchain_cohere.react_multi_hop.parsing import parse_actions
from tests.unit_tests.react_multi_hop import ExpectationType, read_expectation_from_file


@pytest.mark.parametrize(
    "scenario_name, expected_plan, expected_actions, expected_error",
    [
        pytest.param(
            "plan_with_action_normal",
            "Do a thing.\nAnd then do another thing.",
            [
                {"parameters": {"arg1": "value1", "arg2": 2}, "tool_name": "tool1"},
                {"parameters": {"arg3": "value3", "arg4": True}, "tool_name": "tool2"},
            ],
            None,
            id="plan with action (normal)",
        ),
        pytest.param(
            "reflection_with_action_normal",
            "I found out a thing.\nAnd then do another thing.",
            [
                {"parameters": {"arg1": "value1", "arg2": 2}, "tool_name": "tool1"},
            ],
            None,
            id="plan with reflection (normal)",
        ),
        pytest.param(
            "action_only_abnormal",
            "",
            [
                {"parameters": {"arg1": "value1", "arg2": 2}, "tool_name": "tool1"},
            ],
            None,
            id="action only (abnormal)",
        ),
        pytest.param(
            "not_a_plan_reflection_or_action",
            "",
            [],
            ValueError,
            id="invalid generation (abnormal)",
        ),
    ],
)
def test_parse_actions(
    scenario_name: str,
    expected_plan: str,
    expected_actions: List[Dict],
    expected_error: Optional[Any],
) -> None:
    completion = read_expectation_from_file(ExpectationType.completions, scenario_name)

    if expected_error:
        with pytest.raises(expected_error):
            parse_actions(generation=completion)
    else:
        actual_completion, actual_plan, actual_actions = parse_actions(
            generation=completion
        )

        assert completion == actual_completion
        assert expected_plan == actual_plan
        assert expected_actions == actual_actions
