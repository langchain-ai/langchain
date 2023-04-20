import math
import faiss
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from termcolor import colored

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseLanguageModel, Document, BaseMemory
from langchain.vectorstores import FAISS
from langchain.experimental.generative_agents.memory import GenerativeAgentsMemory
class GenerativeAgent(BaseModel):
    name: str
    age: int
    traits: str
    status: str
    """The traits of the character you wish not to change."""
    """A character with memory and innate characteristics."""
    memory: GenerativeAgentsMemory
    llm: BaseLanguageModel
    verbose: bool = False

    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int = 3600  #: :meta private:
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    daily_summaries: List[str]  # : :meta private:

    def chain(self, prompt: PromptTemplate):
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # LLM-related methods

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(entity=entity_name, observation=observation).strip()

    # Memory-related methods
    def _compute_agent_summary(self):
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return self.chain(prompt).run(
            name=self.name,
            queries=[f"{self.name}'s core characteristics"]
        ).strip()

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template("""
{q1}?
Context from memory:
{relevant_memories}
Relevant context: 
""")
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt).run(q1=q1, queries=[q1, q2]).strip()

    def _generate_reaction(
        self,
        observation: str,
        suffix: str
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n" + suffix
        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs = dict(agent_summary_description=agent_summary_description,
                      current_time=current_time_str,
                      relevant_memories=relevant_memories_str,
                      agent_name=self.name,
                      observation=observation,
                      agent_status=self.status)
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs))
        kwargs["most_recent_memories_token"] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]

    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        # AAA
        self.memory.save_context({}, {
            "add_memory": f"{self.name} observed {observation} and reacted by {result}"
        })
        if "REACT:" in result:
            reaction = result.split("REACT:")[-1].strip()
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = result.split("SAY:")[-1].strip()
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            'What would {agent_name} say? To end the conversation, write: GOODBYE: "what to say". Otherwise to continue the conversation, write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        if "GOODBYE:" in result:
            farewell = result.split("GOODBYE:")[-1].strip()
            # AAA
            self.memory.save_context({}, {
                "add_memory": f"{self.name} observed {observation} and said {farewell}"
            })
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = result.split("SAY:")[-1].strip()
            # AAA
            self.memory.save_context({}, {
                "add_memory": f"{self.name} observed {observation} and said {response_text}"
            })
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    def get_summary(self, force_refresh: bool = False) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now()
        since_refresh = (current_time - self.last_refreshed).seconds
        if not self.summary or since_refresh >= self.summary_refresh_seconds or force_refresh:
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        return (
            f"Name: {self.name} (age: {self.age})"
            + f"\nInnate traits: {self.traits}"
            + f"\n{self.summary}"
        )

    def get_full_header(self, force_refresh: bool = False) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        summary = self.get_summary(force_refresh=force_refresh)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        return f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
