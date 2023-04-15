from langchain.auto_agents.autogpt.prompt import AutoGPTPrompt
from langchain.chat_models.base import BaseChatModel
from langchain.tools.base import BaseTool
from langchain.chains.llm import LLMChain
from langchain.agents.agent import AgentOutputParser
from typing import List, Optional
from langchain.schema import SystemMessage, Document

from langchain.vectorstores.base import VectorStoreRetriever
from langchain.auto_agents.autogpt.output_parser import AutoGPTOutputParser


class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        full_message_history: The full message history.
        next_action_count: The number of actions to execute.
        prompt: The prompt to use.
        user_input: The user input.
    """
    def __init__(self,
                 ai_name,
                 memory: VectorStoreRetriever,
                 chain: LLMChain,
                 output_parser: AgentOutputParser,
                 tools: List[BaseTool],
                 ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools

    @classmethod
    def from_llm_and_tools(cls,     ai_name: str,
    ai_role: str,
                           memory: VectorStoreRetriever,
    tools: List[BaseTool], llm: BaseChatModel, output_parser: Optional[AgentOutputParser] = None):
        prompt = AutoGPTPrompt(ai_name=ai_name, ai_role=ai_role, tools=tools, input_variables=["memory", "messages", "goals"])
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools
        )


    def run(self, goals: List[str]):
        # Interaction Loop
        loop_count = 0
        while True:
             # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.full_message_history,
                memory=self.memory
            )

            # Print Assistant thoughts
            print(assistant_reply)

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.tool in tools:
                 tool = tools[action.tool]
                 observation = tool.run(action.tool_input)
                 result = f"Command {tool.name} returned: {observation}"
            else:
                result = f"Unknown command '{action.tool}'. Please refer to the 'COMMANDS' list for available commands and only respond in the specified JSON format."


            memory_to_add = (f"Assistant Reply: {assistant_reply} "
                            f"\nResult: {result} "
                             )

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))