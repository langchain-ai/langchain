from __future__ import annotations

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

# set no tools available (generate only)
notoolset = []

# choose fast LLM
llm_chatgpt = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

# conversational agent requires buffer memory
memory1 = ConversationBufferWindowMemory(memory_key="chat_history", 
                                         return_messages=True)

# initialize conversational agent
agent = initialize_agent(notoolset, llm_chatgpt,
                         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                         verbose=False, memory=memory1)

#
# Create and return Python code, verify generated code contains python header
#
prompt1 ='''Generate a Python class with embedded unit test program that calculates 
the first 100 Fibonaci numbers and prints them out.
Just instantiate the class and test it by running the code
after the class is defined)'''

response1 = agent.run(input=prompt1)
assert "'python" not in response1

#
# Create and return JavaScript code, verify generated code contains javascript header
#
prompt2 ="Generate JavaScript code that calculates first 10 Fibonaci numbers."

response2 = agent.run(input=prompt2)
assert "'javascript" not in response1
