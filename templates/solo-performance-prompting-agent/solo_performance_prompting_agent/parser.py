from langchain_core.agents import AgentAction, AgentFinish


def parse_output(message: str):
    FINAL_ANSWER_ACTION = "<final_answer>"
    includes_answer = FINAL_ANSWER_ACTION in message
    if includes_answer:
        answer = message.split(FINAL_ANSWER_ACTION)[1].strip()
        if "</final_answer>" in answer:
            answer = answer.split("</final_answer>")[0].strip()
        return AgentFinish(return_values={"output": answer}, log=message)
    elif "</tool>" in message:
        tool, tool_input = message.split("</tool>")
        _tool = tool.split("<tool>")[1]
        _tool_input = tool_input.split("<tool_input>")[1]
        if "</tool_input>" in _tool_input:
            _tool_input = _tool_input.split("</tool_input>")[0]
        return AgentAction(tool=_tool, tool_input=_tool_input, log=message)
