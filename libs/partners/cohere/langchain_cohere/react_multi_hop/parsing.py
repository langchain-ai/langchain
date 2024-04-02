import json
import logging
import re
from typing import Dict, List, Tuple, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseOutputParser


class CohereToolsReactAgentOutputParser(
    BaseOutputParser[Union[List[AgentAction], AgentFinish]]
):
    """Parses a message into agent actions/finish."""

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        # Parse the structured output of the final answer.
        if "Answer: " in text:
            prefix_map = {
                "answer": "Answer:",
                "grounded_answer": "Grounded answer:",
                "relevant_docs": "Relevant Documents:",
                "cited_docs": "Cited Documents:",
            }
            parsed_answer = parse_answer_with_prefixes(text, prefix_map)
            return AgentFinish({"output": parsed_answer["answer"]}, text)
        elif any([x in text for x in ["Plan: ", "Reflection: ", "Action: "]]):
            completion, plan, actions = parse_actions(text)
            agent_actions: List[AgentAction] = []
            for i, action in enumerate(actions):
                agent_action = AgentActionMessageLog(
                    tool=action["tool_name"],
                    tool_input=action["parameters"],
                    log=f"\n{action}\n" if i > 0 else f"\n{plan}\n{action}\n",
                    message_log=[AIMessage(content=completion)],
                )
                agent_actions.append(agent_action)

            return agent_actions
        else:
            raise ValueError(
                "\nCould not parse generation as it did not contain Plan, Reflection,"
                + f"Action, or Answer. Input: {text}\n\n"
            )


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
            regex = r"^(Plan|Reflection)\s*\d*\s*:(.*?)(Action\s*\d*\s*:\s*\d*\s*```json\n.*?```)"  # noqa: E501
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
