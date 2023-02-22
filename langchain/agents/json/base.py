"""Agent for interacting with JSON."""
import string
from typing import Any, List, Optional, Sequence

from langchain.agents.agent import Agent
from langchain.agents.mrkl.base import ZeroShotAgent, create_zero_shot_prompt
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.json.prompt import PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool

class JsonAgent(ZeroShotAgent):
    