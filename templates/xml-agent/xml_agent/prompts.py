from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = """You are a helpful assistant. Help the user answer any questions.

You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

When you are done, you can respond as normal to the user.

Example 1:

Human: Hi!

Assistant: Hi! How are you?

Human: What is the weather in SF?
Assistant: <tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>
It is 64 degress in SF


Begin!"""  # noqa: E501

conversational_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
        ("ai", "{agent_scratchpad}"),
    ]
)


def parse_output(message):
    text = message.content
    if "</tool>" in text:
        tool, tool_input = text.split("</tool>")
        _tool = tool.split("<tool>")[1]
        _tool_input = tool_input.split("<tool_input>")[1]
        if "</tool_input>" in _tool_input:
            _tool_input = _tool_input.split("</tool_input>")[0]
        return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
    else:
        return AgentFinish(return_values={"output": text}, log=text)
