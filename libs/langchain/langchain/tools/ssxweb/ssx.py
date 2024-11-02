from openai import OpenAI

#KEY
K_OpenAI = "sk-proj- ****"

# LIBS: LANGCHAIN, openAI...
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

#import dspy
import openai
from openai import OpenAI
#import logging
#logging
import os
os.environ["OPENAI_API_KEY"] = K_OpenAI
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0'

# setup SECRET KEY OPENAI
secretk = os.getenv("OPENAI_API_KEY")
#client = OpenAI(api_key=secretk)
openai.api_key = secretk
client = OpenAI(api_key=secretk)

# Configuring the language model (e.g., GPT-4)
llm = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=2048)
# LET'S SWITCH TO SESSION...

# Function to use the retriever as a tool
@tool
def ssearx_web(query: str) -> Tool:
    """
        A web search tool that automates Google searches and accesses specific websites to scrape their content.
        This tool retrieves relevant links, accesses them, and extracts the content as required.

        Capabilities:
            - Perform Google searches for specific topics or queries.
            - Directly access websites from the search results and extract their content.
            - Summarize the extracted content or provide relevant details based on the user query.

        Important: This tool adheres to legal restrictions, avoiding searches or content related to prohibited topics under Brazilian law.

        Functions:
            * ssx.ssxGoogle(query): Executes a Google search based on the user query and retrieves relevant links.
            * ssx.ssxWebsite(link): Accesses the specified link and extracts content from it.

        Args:
            query (str): The search phrase or URL to search or access.

        Returns:
            str: A summary or detailed information extracted from the target website or search results.
    """

    from simplesearx import SSearX
    ssx = SSearX()

    if "google":
        result = ssx.ssxGoogle(query)
    elif "website":
        result = ssx.ssxWebsite(query)
    else:
        result = "Please clarify the type of search you want to perform."
    
    return result
    
##### END OF TOOLBOX #########################################################

##### CONNECTION TO THE TOOLBOX, CREATING CHAT_HISTORY LIST AND FUNCTION TO EXTEND CHAT :) ...
tools = [ssearx_web]
llmtools = llm.bind_tools(tools)

def agenteIA(usuario_prompt):
    """
        The main function to handle user queries using the AI agent with web search capabilities.

        Parameters:
            user_prompt (str): The user's input or query.

        Returns:
            str: The agent's response based on the processed query.
    """

    #### CHATPROMPT TEMPLATE - WHO THE AGENT IS AND HOW IT SHOULD BEHAVE, CHAT MEMORY SESSION #############################
    prompt = ChatPromptTemplate.from_messages([
         "system", """This is an advanced web search tool with capabilities to retrieve important content, promotions, contact information, prices, and more. 
         It can perform Google searches to gather relevant links and directly access websites to scrape detailed information as needed,
         providing comprehensive and accurate responses based on user queries.""",
         "user", usuario_prompt
    ])

    #### AGENT EXECUTOR THAT WILL OPERATE WITH THE TOOLS ############################
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )    }
        | prompt
        | llmtools
        | OpenAIToolsAgentOutputParser()
        
    )
    #### AGENT EXECUTOR THAT WILL OPERATE WITH THE TOOLS ############################
    agent_executor = AgentExecutor(
        tools = tools,
        llm = llm,
        agent = agent,  # Agent type that can decide on tool usage
        verbose = True, 
        max_iterations=1
    )

    try:
        result__ = agent_executor.invoke({"input": usuario_prompt}, v=True)
        return result__['output']
    except Exception as e:
        print(f"ERROR IN INVOKE: {e}")    


agente = agenteIA("consulting the grupo portfolio website through google")
print(agente)
