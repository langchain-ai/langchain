import json
import logging
import re
from typing import Any, Dict, List, Mapping, Tuple, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseOutputParser

from langchain_cohere import CohereCitation

OUTPUT_KEY = "output"
GROUNDED_ANSWER_KEY = "grounded_answer"


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
            return AgentFinish(
                return_values={
                    OUTPUT_KEY: parsed_answer["answer"],
                    GROUNDED_ANSWER_KEY: parsed_answer["grounded_answer"],
                },
                log=text,
            )
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


def parse_citations(
    grounded_answer: str, documents: List[Mapping]
) -> Tuple[str, List[CohereCitation]]:
    """
    Parses a grounded_generation (from parse_actions) and documents (from
    convert_to_documents) into a (generation, CohereCitation list) tuple.
    """

    no_markup_answer, parsed_answer = _parse_answer_spans(grounded_answer)
    citations: List[CohereCitation] = []
    start = 0

    for answer in parsed_answer:
        text = answer.get("text", "")
        document_indexes = answer.get("cited_docs")
        if not document_indexes:
            # There were no citations for this piece of text.
            start += len(text)
            continue
        end = start + len(text)

        # Look up the cited document by index
        cited_documents: List[Mapping] = []
        for index in set(document_indexes):
            if index >= len(documents):
                # The document index doesn't exist
                continue
            cited_documents.append(documents[index])

        citations.append(
            CohereCitation(
                start=start,
                end=end,
                text=text,
                documents=cited_documents,
            )
        )
        start = end

    return no_markup_answer, citations


def _strip_spans(answer: str) -> str:
    """removes any <co> tags from a string, including trailing partial tags

    input: "hi my <co>name</co> is <co: 1> patrick</co:3> and <co"
    output: "hi my name is patrick and"

    Args:
        answer (str): string

    Returns:
        str: same string with co tags removed
    """
    answer = re.sub(r"<co(.*?)>|</co(.*?)>", "", answer)
    idx = answer.find("<co")
    if idx > -1:
        answer = answer[:idx]
    idx = answer.find("</")
    if idx > -1:
        answer = answer[:idx]
    return answer


def _parse_answer_spans(grounded_answer: str) -> Tuple[str, List[Dict[str, Any]]]:
    actual_cites = []
    for c in re.findall(r"<co:(.*?)>", grounded_answer):
        actual_cites.append(c.strip().split(","))
    no_markup_answer = _strip_spans(grounded_answer)

    current_idx = 0
    parsed_answer: List[Dict[str, Union[str, List[int]]]] = []
    cited_docs_set = []
    last_entry_is_open_cite = False
    parsed_current_cite_document_idxs: List[int] = []

    while current_idx < len(grounded_answer):
        current_cite = re.search(r"<co: (.*?)>", grounded_answer[current_idx:])
        if current_cite:
            # previous part
            parsed_answer.append(
                {
                    "text": grounded_answer[
                        current_idx : current_idx + current_cite.start()
                    ]
                }
            )

            current_cite_document_idxs = current_cite.group(1).split(",")
            parsed_current_cite_document_idxs = []
            for cited_idx in current_cite_document_idxs:
                if cited_idx.isdigit():
                    cited_idx = int(cited_idx.strip())
                    parsed_current_cite_document_idxs.append(cited_idx)
                    if cited_idx not in cited_docs_set:
                        cited_docs_set.append(cited_idx)

            current_idx += current_cite.end()

            current_cite_close = re.search(
                r"</co: " + current_cite.group(1) + ">", grounded_answer[current_idx:]
            )

            if current_cite_close:
                # there might have been issues parsing the ids, so we need to check
                # that they are actually ints and available
                if len(parsed_current_cite_document_idxs) > 0:
                    pt = grounded_answer[
                        current_idx : current_idx + current_cite_close.start()
                    ]
                    parsed_answer.append(
                        {"text": pt, "cited_docs": parsed_current_cite_document_idxs}
                    )
                else:
                    parsed_answer.append(
                        {
                            "text": grounded_answer[
                                current_idx : current_idx + current_cite_close.start()
                            ],
                        }
                    )

                current_idx += current_cite_close.end()

            else:
                last_entry_is_open_cite = True
                break
        else:
            break

    # don't forget about the last one
    if last_entry_is_open_cite:
        pt = _strip_spans(grounded_answer[current_idx:])
        parsed_answer.append(
            {"text": pt, "cited_docs": parsed_current_cite_document_idxs}
        )
    else:
        parsed_answer.append({"text": _strip_spans(grounded_answer[current_idx:])})
    return no_markup_answer, parsed_answer
