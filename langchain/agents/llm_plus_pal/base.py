"""Chain that use PAL when doing math."""
from typing import Any, ClassVar, List, Optional, Tuple

from langchain.agents.agent import Agent
from langchain.agents.llm_plus_pal.prompt import PROMPT
from langchain.agents.tools import Tool
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate
from langchain.serpapi import SerpAPIWrapper
from langchain import OpenAI
from langchain.chains.pal.base import PALChain

PAL_FOLLOWUP_PREFIX = "Follow up with PAL: "
PAL_ANSWER_PREFIX = "PAL return: "
PAL_TOOL_NAME = "PAL"
FINISH_PREFIX = "So the final answer is:"
FINAL_ANSWER_TOOL = "Final Answer"

class LLMPlusPalAgent(Agent):
    """Agent for using PAL as needed ."""

    prompt: ClassVar[BasePromptTemplate] = PROMPT

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        if len(tools) != 1:
            raise ValueError(f"Exactly one tool must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Intermediate Answer"}:
            raise ValueError(
                f"Tool name should be Intermediate Answer, got {tool_names}"
            )

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        last_line = text.split("\n")[-1]
            
        if PAL_FOLLOWUP_PREFIX in last_line:
            after_thinkslow_prefix = text.split(PAL_FOLLOWUP_PREFIX)[-1]
            return PAL_TOOL_NAME, after_thinkslow_prefix        
        elif FINISH_PREFIX in last_line:
            return FINAL_ANSWER_TOOL, last_line[len(FINISH_PREFIX):]
        
        return None

    def _fix_text(self, text: str) -> str:
        return f"{text}\nSo the final answer is:"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return PAL_ANSWER_PREFIX

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return ""

    # @property
    # def starter_string(self) -> str:
    #     """Put this string after user input but before first LLM call."""
    #     return ""


class LLMPlusPalChain(LLMPlusPalAgent):
    """Chain that does calculation on PAL.

    Example:
        .. code-block:: python

    """

    def __init__(self, llm: LLM, **kwargs: Any):
        """Initialize with just an LLM and a pal chain."""
        pal = OpenAI(model_name='text-davinci-003', temperature=0., max_tokens=512)
        pal_chain = PALChain.from_math_prompt(pal, verbose=True)        
        pal_tool = Tool(name=PAL_TOOL_NAME, func=pal_chain.run)
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        super().__init__(llm_chain=llm_chain, tools=[pal_tool], **kwargs)
