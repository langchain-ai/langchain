from typing import Optional, Type


from pydantic import BaseModel
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.networkx_graph import KG_TRIPLE_DELIMITER
from langchain_community.graphs.networkx_graph import parse_triples

# flake8: noqa

_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the text."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property.\n\n"
    "EXAMPLE\n"
    "It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "{text}"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)


class GraphIndexCreator(BaseModel):
    """Functionality to create graph index."""

    llm: Optional[BaseLanguageModel] = None
    graph_type: Type[NetworkxEntityGraph] = NetworkxEntityGraph

    def from_text(
        self, text: str, prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    ) -> NetworkxEntityGraph:
        """Create graph index from text."""
        if self.llm is None:
            raise ValueError("llm should not be None")
        graph = self.graph_type()
        # Temporary local scoped import while community does not depend on
        # langchain explicitly
        try:
            from langchain.chains import LLMChain
        except ImportError:
            raise ImportError(
                "Please install langchain to use this functionality. "
                "You can install it with `pip install langchain`."
            )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.predict(text=text)
        knowledge = parse_triples(output)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph

    async def afrom_text(
        self, text: str, prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    ) -> NetworkxEntityGraph:
        """Create graph index from text asynchronously."""
        if self.llm is None:
            raise ValueError("llm should not be None")
        graph = self.graph_type()
        # Temporary local scoped import while community does not depend on
        # langchain explicitly
        try:
            from langchain.chains import LLMChain
        except ImportError:
            raise ImportError(
                "Please install langchain to use this functionality. "
                "You can install it with `pip install langchain`."
            )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = await chain.apredict(text=text)
        knowledge = parse_triples(output)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph
