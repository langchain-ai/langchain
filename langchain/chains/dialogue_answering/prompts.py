from langchain.prompts.prompt import PromptTemplate


SUMMARY_TEMPLATE = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=SUMMARY_TEMPLATE
)

DIALOGUE_PREFIX = """Have a conversation with a human,Analyze the content of the conversation.
You have access to the following tools: """
DIALOGUE_SUFFIX = """Begin!

{chat_history}
Question: {input}
{agent_scratchpad}"""
