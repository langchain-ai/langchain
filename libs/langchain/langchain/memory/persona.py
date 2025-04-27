from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
import re

from langchain_core.memory import BaseMemory


class EnrichedMessage(BaseModel):
    """A message enriched with persona traits and metadata."""

    id: str
    content: str
    traits: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PersonaMemory(BaseMemory):
    """
    Memory module that dynamically tracks emotional and behavioral traits
    exhibited by an agent over the course of a conversation.

    Traits are automatically detected from output messages and
    accumulated across interactions. Recent messages and detected traits
    can be retrieved to enrich future prompts or modify agent behavior.
    """

    memory_key: str = "persona"
    input_key: str = "input"
    output_key: str = "output"
    k: int = 10
    trait_detection_engine: Optional[Callable[[str], Dict[str, int]]] = None
    persona_traits: List[str] = Field(default_factory=list)
    recent_messages: List[Any] = Field(default_factory=list)

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def clear(self) -> None:
        """Clear the memory state."""
        self.persona_traits = []
        self.recent_messages = []

    def _detect_traits(self, text: str) -> Dict[str, int]:
        """
        Detect persona traits using both a default method and optionally an external engine.

        Always guarantees a usable result, even if external services fail. (Just lower quality as hard-coded)
        """
        # Always run the fast, local simple detection first
        trait_patterns = {
            "apologetic": ["sorry", "apologize", "apologies", "I apologize"],
            "enthusiastic": ["!", "awesome", "great job", "fantastic"],
            "formal": ["Dear", "Sincerely", "Respectfully"],
            "cautious": ["maybe", "perhaps", "I think", "it could be"],
            "hesitant": ["maybe", "...", "I'm not sure", "perhaps", "unsure"],
            "friendly": ["Hi", "Hey", "Hello", "Good to see you", "friendly"],
            "curious": ["?", "I wonder", "Could you"],
            "analytical": ["analytical", "analyze", "analysis", "logical", "reasoning"],
        }

        trait_hits: Dict[str, int] = {}
        lowered_text = text.lower()

        # Clean the text for word matching by removing punctuation
        clean_text = re.sub(r"[^\w\s]", " ", lowered_text)
        words = clean_text.split()

        for trait, patterns in trait_patterns.items():
            count = 0
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if trait == "friendly":
                    if lowered_text.startswith(pattern_lower) or pattern_lower in words:
                        count += 1
                elif trait == "enthusiastic":
                    if pattern == "!":
                        count = lowered_text.count("!")
                else:
                    if pattern_lower in lowered_text:
                        count += 1

            if count > 0:
                trait_hits[trait] = count

        # Now attempt external engine if available
        if self.trait_detection_engine:
            try:
                external_traits = self.trait_detection_engine(text)
                if isinstance(external_traits, dict):
                    return external_traits
            except Exception:
                pass  # Silently fall back to default detection

        # Fall back to simple default detection if external fails or is unavailable
        return trait_hits

    def load_memory_variables(
        self, inputs: Dict[str, Any], include_messages: bool = False
    ) -> Dict[str, Any]:
        """Return the stored persona traits and optionally recent conversation messages."""

        memory_data = {"traits": self.persona_traits.copy()}

        if include_messages:
            memory_data["recent_messages"] = [
                message.model_dump() for message in self.recent_messages
            ]

        return {self.memory_key: memory_data}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Analyze outputs and update the persona traits and recent messages."""

        input_text = inputs.get(self.input_key, "")
        output_text = outputs.get(self.output_key, "")

        traits_detected = self._detect_traits(output_text)

        message = EnrichedMessage(
            id=str(len(self.recent_messages) + 1),
            content=output_text,
            traits=list(traits_detected.keys()),
            metadata={"traits_count": traits_detected},
        )

        self.recent_messages.append(message)

        # Trim messages to maintain memory size k
        if len(self.recent_messages) > self.k:
            self.recent_messages = self.recent_messages[-self.k :]

        # Update persona traits
        self.persona_traits = list(
            {trait for msg in self.recent_messages for trait in msg.traits}
        )
