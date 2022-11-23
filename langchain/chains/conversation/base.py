"""Chain that carries on a conversation and calls an LLM."""
from pydantic import BaseModel, Extra, Field

from langchain.chains.base import Memory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.prompt import PROMPT
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate


class ConversationChain(LLMChain, BaseModel):
    """Chain to have a conversation and load context from memory.

    Example:
        .. code-block:: python

            from langchain import ConversationChain, OpenAI
            conversation = ConversationChain(llm=OpenAI())
    """

    memory: Memory = Field(default_factory=ConversationBufferMemory)
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""

    input_key: str = "input"  #: :meta private:
    output_key: str = "response"  #: :meta private:
    buffer: str = ""  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
