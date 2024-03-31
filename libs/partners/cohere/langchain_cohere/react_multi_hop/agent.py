"""
    Cohere multi-hop agent enables multiple tools to be used in sequence to complete
    your task.

    This agent uses a multi hop prompt by Cohere, which is experimental and subject
    to change. The latest prompt can be used by upgrading the langchain-cohere package.
"""
from typing import List, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain_cohere.react_multi_hop.parsing import (
    parse_actions,
    parse_answer_with_prefixes,
)
from langchain_cohere.react_multi_hop.prompt import (
    multi_hop_prompt,
)


def create_cohere_react_agent(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    prompt: ChatPromptTemplate,
) -> Runnable:
    agent = (
        RunnablePassthrough.assign(
            # Handled in the multi_hop_prompt
            agent_scratchpad=lambda _: [],
        )
        | multi_hop_prompt(tools=tools, prompt=prompt)
        | llm.bind(stop=["\nObservation:"], raw_prompting=True)
        | CohereToolsReactAgentOutputParser()
    )
    return agent


class CohereToolsReactAgentOutputParser(
    BaseOutputParser[Union[List[AgentAction], AgentFinish]]
):
    """Parses a message into agent actions/finish."""

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        # Parse the structured output of the final answer.
        if "Answer: " in text:
            prefix_map = {
                "answer": "Answer:",
                "grounded_answer": "Grounded answer:",
                "relevant_docs": "Relevant Documents:",
                "cited_docs": "Cited Documents:",
            }
            parsed_answer = parse_answer_with_prefixes(text, prefix_map)
            return AgentFinish({"output": parsed_answer["answer"]}, text)
        elif any([x in text for x in ["Plan: ", "Reflection: ", "Action: "]]):
            completion, plan, actions = parse_actions(text)
            agent_actions: List[AgentAction] = []
            for i, action in enumerate(actions):
                agent_action = AgentActionMessageLog(
                    tool=action["tool_name"],
                    tool_input=action["parameters"],
                    log=f"\n{action}\n" if i > 0 else f"\n{plan}\n{action}\n",
                    message_log=[AIMessage(content=completion)],
                )
                agent_actions.append(agent_action)

            return agent_actions
        else:
            raise ValueError(
                "\nCould not parse generation as it did not contain Plan, Reflection,"
                + f"Action, or Answer. Input: {text}\n\n"
            )
