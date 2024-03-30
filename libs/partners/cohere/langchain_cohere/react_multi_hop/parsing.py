import json
import logging
import re
from typing import Dict, List, Tuple


def parse_jsonified_tool_use_generation(
    tool_use_generation: str, tool_use_prefix: str
) -> List[Dict]:
    """Parses model-generated jsonified actions.

    Expects input of the form
    "{tool_use_prefix}: ```json\n[{list of jsonified objects}]```"

    outputs parsed list of jsonified objects.
    """

    def _extract_codeblocks_from_md(text: str) -> List[str]:
        return re.findall(r"`{3}([^`]*)`{0,3}", text)

    raw_generation = re.sub(f"^{tool_use_prefix} ", "", tool_use_generation)
    code_block_sections = _extract_codeblocks_from_md(raw_generation)

    if len(code_block_sections) != 1:  # should have exactly 1 code block
        raise ValueError(f"Action Parsing Failed: {tool_use_generation}")
    # only json allowed:
    assert code_block_sections[0].startswith(
        "json\n"
    ), f"Action Parsing Failed: {tool_use_generation}"

    actions = json.loads(re.sub("^json\n", "", code_block_sections[0]))

    if not isinstance(actions, list):
        raise ValueError(f"Action Parsing Failed: {tool_use_generation}")

    if len(actions):
        if any(
            not isinstance(action, Dict) or "tool_name" not in action
            for action in actions
        ):
            raise ValueError(f"Action Parsing Failed: {tool_use_generation}")
    return actions


def parse_answer_with_prefixes(
    completion: str, prefixes: Dict[str, str]
) -> Dict[str, str]:
    """parses string into key-value pairs,
       according to patterns supplied in prefixes. Also strips.

    if inputs are:
        completion = "\nhello: sam\ngoodbye then: paul.",
        prefixes = {"greeting": "hello:", "farewell": "goodbye then:"}

    the expected returned result is:
        {"greeting": "sam", "farewell": "paul."}

    Args:
        completion (str): text to split
        prefixes (Dict[str, str]): a key-value dict of keys and patterns.
        See example above

    Returns:
        Dict[str, str]: parsed result
    """
    # sort out prefixes
    re_pat = "(" + "|".join([re.escape(p) for p in prefixes.values()]) + ")"
    reverse_prefix_map = {v: k for k, v in prefixes.items()}
    split = re.split(re_pat, completion)
    split = split[1:]
    parsed = {}
    for prefix, value in zip(split[::2], split[1::2]):
        if prefix in reverse_prefix_map:  # if the prefix is a match
            if (
                reverse_prefix_map[prefix] not in parsed
            ):  # first occurrence of a prefix is kept, others discarded
                parsed[reverse_prefix_map[prefix]] = value.strip()
    return parsed


def parse_actions(generation: str) -> Tuple[str, str, List[Dict]]:
    """Parse action selections from model output."""
    plan = ""
    actions = generation
    try:
        if "Plan: " in generation or "Reflection: " in generation:
            # Model is trained to output a Plan or Reflection followed by an action.
            # Use regex to extract the plan and action.
            regex = (
                r"^(Plan|Reflection)\s*\d*\s*:(.*?)"
                + r"(Action\s*\d*\s*:\s*\d*\s*```json\n.*?```)"
            )
            action_match = re.search(regex, generation, re.DOTALL)
            if not action_match:
                raise ValueError(
                    f"Failed to parse multihop completion for input: {generation}"
                )
            plan = action_match.group(2).strip()
            actions = action_match.group(3).strip()
        else:
            # Catch the case where model outputs only an action.
            regex = r"^(Action\s*\d*\s*:\s*\d*\s*```json\n.*?```)"
            action_match = re.search(regex, generation, re.DOTALL)
            if not action_match:
                raise ValueError(
                    f"Failed to parse multihop completion for input: {generation}"
                )
            actions = action_match.group(1).strip()
    except Exception as e:
        logging.error(f"Failed to parse multihop completion for input: {generation}")
        logging.error(f"Error: {e}")

    parsed_actions = parse_jsonified_tool_use_generation(actions, "Action:")
    return generation, plan, parsed_actions
