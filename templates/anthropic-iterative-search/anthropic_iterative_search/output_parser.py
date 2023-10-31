import re

from langchain.schema.agent import AgentAction, AgentFinish

from .agent_scratchpad import _format_docs


def extract_between_tags(tag: str, string: str, strip: bool = True) -> str:
    ext_list = re.findall(f"<{tag}\s?>(.+?)</{tag}\s?>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    if ext_list:
        if len(ext_list) != 1:
            raise ValueError
        # Only return the first one
        return ext_list[0]


def parse_output(outputs):
    partial_completion = outputs["partial_completion"]
    steps = outputs["intermediate_steps"]
    search_query = extract_between_tags(
        "search_query", partial_completion + "</search_query>"
    )
    if search_query is None:
        docs = []
        str_output = ""
        for action, observation in steps:
            docs.extend(observation)
            str_output += action.log
            str_output += "</search_query>" + _format_docs(observation)
        str_output += partial_completion
        return AgentFinish({"docs": docs, "output": str_output}, log=partial_completion)
    else:
        return AgentAction(
            tool="search", tool_input=search_query, log=partial_completion
        )
