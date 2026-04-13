"""Muninn memory integration for LangChain."""

from typing import Any, Dict, List, Optional

from langchain_core.memory import BaseMemory
from langchain_core.messages import HumanMessage, AIMessage
from muninn import MuninnClient


class MuninnMemory(BaseMemory):
    """
    LangChain memory backed by Muninn semantic search.
    
    Provides persistent, searchable memory for LangChain agents.
    99.1% accuracy on LOCOMO benchmark.
    
    Example:
        from langchain_muninn import MuninnMemory
        from langchain.agents import initialize_agent
        
        memory = MuninnMemory(api_key="muninn_xxx")
        agent = initialize_agent(tools=tools, llm=llm, memory=memory)
    
    Args:
        api_key: Muninn API key
        organization_id: Organization ID for multi-tenant isolation
        base_url: Muninn API base URL (default: https://api.muninn.au)
        memory_type: Memory type for stored memories (default: conversational)
    """
    
    api_key: str
    organization_id: str = "default"
    base_url: str = "https://api.muninn.au"
    memory_type: str = "conversational"
    
    _client: Optional[MuninnClient] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self._client = MuninnClient(
            api_key=self.api_key,
            organization_id=self.organization_id,
            base_url=self.base_url
        )
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return ["history"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory from Muninn using semantic search."""
        query = str(inputs)
        
        memories = self._client.search(
            query=query,
            limit=10,
            search_type="hybrid"
        )
        
        # Format as conversation history
        history = []
        for memory in memories:
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})
            role = metadata.get("role", "user")
            
            if role == "assistant":
                history.append(AIMessage(content=content))
            else:
                history.append(HumanMessage(content=content))
        
        return {"history": history}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation turn to Muninn."""
        user_input = str(inputs)
        if user_input:
            self._client.store(
                content=user_input,
                memory_type=self.memory_type,
                metadata={"role": "user", "source": "langchain"}
            )
        
        ai_output = outputs.get("output", str(outputs))
        if ai_output:
            self._client.store(
                content=ai_output,
                memory_type=self.memory_type,
                metadata={"role": "assistant", "source": "langchain"}
            )
    
    def clear(self) -> None:
        """Clear memory (no-op for safety)."""
        pass


class MuninnEntityMemory(BaseMemory):
    """
    Entity-focused memory for LangChain agents.
    
    Stores and retrieves facts about entities mentioned in conversation.
    Best for agents that need to remember specific information about
    people, organizations, or concepts.
    
    Example:
        from langchain_muninn import MuninnEntityMemory
        
        memory = MuninnEntityMemory(api_key="muninn_xxx")
        # Automatically extracts: "James works at TechCorp"
        # into entity facts: {James: {works_at: TechCorp}}
    """
    
    api_key: str
    organization_id: str = "default"
    base_url: str = "https://api.muninn.au"
    
    _client: Optional[MuninnClient] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self._client = MuninnClient(
            api_key=self.api_key,
            organization_id=self.organization_id,
            base_url=self.base_url
        )
    
    @property
    def memory_variables(self) -> List[str]:
        return ["entity_facts"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load facts about entities mentioned in inputs."""
        query = str(inputs)
        
        facts = self._client.search(
            query=query,
            limit=20,
            search_type="hybrid"
        )
        
        # Group by entity
        entity_facts = {}
        for fact in facts:
            content = fact.get("content", "")
            entities = fact.get("entities", [])
            
            for entity in entities:
                if entity not in entity_facts:
                    entity_facts[entity] = []
                entity_facts[entity].append(content)
        
        return {"entity_facts": entity_facts}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Store conversation turn (Muninn handles entity extraction)."""
        combined = f"User: {inputs}\nAssistant: {outputs}"
        self._client.store(
            content=combined,
            memory_type="conversational",
            metadata={"source": "langchain_entity"}
        )
    
    def clear(self) -> None:
        pass


__all__ = ["MuninnMemory", "MuninnEntityMemory"]