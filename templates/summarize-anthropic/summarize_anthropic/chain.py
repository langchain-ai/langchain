from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

# Create chain
prompt = hub.pull("hwchase17/anthropic-paper-qa")
model = ChatAnthropic(model="claude-3-sonnet-20240229", max_tokens=4096)
chain = prompt | model | StrOutputParser()
