import re
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.experimental.generative_agents.memory import GenerativeAgentMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel


class GenerativeAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    """The character's name."""

    age: Optional[int] = None
    """The optional age of the character."""
    traits: str = "N/A"
    """Permanent traits to ascribe to the character."""
    status: str
    """The traits of the character you wish not to change."""
    memory: GenerativeAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
    llm: BaseLanguageModel
    """The underlying language model."""
    verbose: bool = False
    summary: str = ""  #: :meta private:
    """Stateful self-summary generated via reflection on the character's memory."""

    summary_refresh_seconds: int = 3600  #: :meta private:
    """How frequently to re-generate the summary."""

    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    """The last time the character's summary was regenerated."""

    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    """Summary of the events in the plan that the agent took."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # LLM-related methods
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            """
Observation: {observation}
What is the observed entity in the observation?
Entity="""
        )
        return self.chain(prompt).run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            """
Observation: {observation}
What is the action {entity} is doing in the observation?
Action={entity} is """
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            """
{q1}?
Context from memory:
{relevant_memories}
Relevant context: """
        )
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()

    def _generate_reaction(self, observation: str, suffix: str) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            f"""
{self.get_summary()}
It is {datetime.now().strftime("%B %d, %Y, %I:%M %p")}.
{self.name}'s status: {self.status}
Summary of relevant context from {self.name}'s memory:
{self.summarize_related_memories(observation)}
Most recent observations: {{most_recent_memories}}
Observation: {observation}

{suffix}"""
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="")
        )
        return (
            self.chain(prompt=prompt).run(recent_memories_token=consumed_tokens).strip()
        )

    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = f"""
Should {self.name} react to the observation, and if so what would be the reaction? 
Respond in one line.
If the action is to engage in dialogue, write:
SAY: "what to say"
otherwise, write:
REACT: {self.name}'s reaction (if anything).
Either do nothing, react, or say something but not both.
"""
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split("\n")[0]
        self.memory.add_memory(
            f"{self.name} observed " f"{observation} and reacted by {result}"
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = f"""
What would {self.name} say? To end the conversation, write:
GOODBYE: "what to say". 
Otherwise to continue the conversation, write: 
SAY: "what to say next"
"""
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {farewell}"
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {response_text}"
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    ######################################################
    # Agent stateful' summary methods.                   #
    # Each dialog or response prompt includes a header   #
    # summarizing the agent's self-description. This is  #
    # updated periodically through probing its memories  #
    ######################################################
    def _compute_agent_summary(self) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            """
How would you summarize {name}'s core characteristics given the following statements:
{relevant_memories_simple}
Do not embellish.

Summary: """
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    def get_summary(self, force_refresh: bool = False) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now()
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        age = self.age if self.age is not None else "N/A"
        return f"""
Name: {self.name} (age: {age})
Innate traits: {self.traits}
{self.summary}
"""

    def get_full_header(self, force_refresh: bool = False) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        summary = self.get_summary(force_refresh=force_refresh)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        return (
            f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
        )
