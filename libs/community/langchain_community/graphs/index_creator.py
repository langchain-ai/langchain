from typing import Optional, Type


from pydantic import BaseModel
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.networkx_graph import KG_TRIPLE_DELIMITER
from langchain_community.graphs.networkx_graph import parse_triples, parse_json_triples

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

_VERTEXAI_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (    
    "<goal>Given a text document, identify ALL (subject, object, predicate) triplets where the predicate is clear based on the surrounding text.</goal>\n\n"
    "<steps>\n"
    "1. Identify all entities in the text document. For each identified entity, extract the following information:\n"
    "\t- entity_name: Name of the entity, capitalized\n"
    "\t- entity_description: A short description of the entity relevant to potential relationships (e.g., location, function)\n\n"
    "2. Analyze the relationships between the identified entities.\n" 
    "\t- For each pair of entities, examine the surrounding text to determine if there's a clear connection between them based on verbs, prepositions, or other clues (e.g., 'build', 'located on').\n"
    "\t- If a relationship is found, extract the following information for the triplet:\n"
    "\t\t- subject: The subject is the entity being described, as identified in step 1\n"
    "\t\t- object: The object is the value of the property, as identified in step 1\n"
    "\t\t- predicate: The predicate is the property of the subject that is being described.\n"
    "</steps>\n\n"
    "<example>\n"
    "<text>A chef cooks delicious food in a restaurant.</text>\n"
    "<output>\n"
    "[\n"
    "\t{{\n"
    "\t\t\"subject\": \"Chef\",\n"
    "\t\t\"object\": \"Food\",\n"
    "\t\t\"predicate\": \"cooks\"\n"
    "\t}},\n"
    "\t{{\n"
    "\t\t\"subject\": \"Chef\",\n"
    "\t\t\"object\": \"Restaurant\",\n"
    "\t\t\"predicate\": \"works in\"\n"
    "\t}}\n"
    "]\n"
    "</output>\n"
    "</example>\n\n"
    "<real_data>\n"
    "<text>\n" 
    "{text}\n"
    "</text>\n\n"
    "<output>\n\n"
)

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "name of the source entity, as identified in step 1"
            },
            "object": {
                "type": "string",
                "description": "name of the target entity, as identified in step 1"
            },
            "predicate": {
                "type": "string",
                "description": "a very brief (2-5 words) description of the relationship"
            },
        },
        "required": ["subject", "object", "predicate"],
    },
}

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

VERTEXAI_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_VERTEXAI_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
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
        # Determin BaseLanguageModel class name. If ChatVertexAI
        # is the BaseLanguageModel, then use the Vertex AI specific prompt
        llm_class_name = self.llm.__class__.__name__
        if llm_class_name == "ChatVertexAI":
            chain = VERTEXAI_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT | self.llm.bind(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA
                )
            output = chain.invoke({"text": text})
            knowledge = parse_json_triples(output.content)
        else:
            chain = prompt | self.llm
            output = chain.invoke({"text": text})
            knowledge = parse_triples(output.content)
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
        llm_class_name = self.llm.__class__.__name__
        if llm_class_name == "ChatVertexAI":
            chain = VERTEXAI_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT | self.llm.bind(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA
                )
            output = await chain.ainvoke({"text": text})
            knowledge = parse_json_triples(output.content)
        else:
            chain = prompt | self.llm
            output = await chain.ainvoke({"text": text})
            knowledge = parse_triples(output.content)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph
