from langchain.agents import AgentExecutor
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

from .agent_scratchpad import format_agent_scratchpad
from .output_parser import parse_output
from .prompts import retrieval_prompt
from .retriever import retriever_description, search

prompt = ChatPromptTemplate.from_messages(
    [
        ("user", retrieval_prompt),
        ("ai", "{agent_scratchpad}"),
    ]
)
prompt = prompt.partial(retriever_description=retriever_description)

model = ChatAnthropic(model="claude-2", temperature=0, max_tokens_to_sample=1000)

chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_agent_scratchpad(x["intermediate_steps"])
    )
    | prompt
    | model.bind(stop_sequences=["</search_query>"])
    | StrOutputParser()
)

agent_chain = (
    RunnableMap(
        {
            "partial_completion": chain,
            "intermediate_steps": lambda x: x["intermediate_steps"],
        }
    )
    | parse_output
)

executor = AgentExecutor(agent=agent_chain, tools=[search], verbose=True)
