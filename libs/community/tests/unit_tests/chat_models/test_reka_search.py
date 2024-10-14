from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain.chat_models import ChatReka
from langchain_core.tools import Tool

from langchain_community.utilities import GoogleSerperAPIWrapper

# Set up API keys
# os.environ["SERPER_API_KEY"] = "your_serper_api_key_here"
load_dotenv()

# Initialize ChatReka
chat_reka = ChatReka(
    model="reka-core",
    temperature=0.4,
)
prompt = hub.pull("hwchase17/self-ask-with-search")
# Initialize Google Serper API Wrapper
search = GoogleSerperAPIWrapper()

# Define tools
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="""
        useful for when you need to ask with search. 
        """,
    )
]

# Initialize the agent
agent = create_self_ask_with_search_agent(chat_reka, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)


# Example usage
if __name__ == "__main__":
    query = "What is the hometown of the reigning men's U.S. Open champion?"
    agent_executor.invoke({"input": "query"})
