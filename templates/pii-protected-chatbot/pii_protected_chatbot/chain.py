from typing import List, Tuple

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough
from presidio_analyzer import AnalyzerEngine


# Formatting for chat history
def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# Prompt we will use
_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who speaks like a pirate",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{text}"),
    ]
)

# Model we will use
_model = ChatOpenAI()

# Standard conversation chain.
chat_chain = (
    {
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "text": lambda x: x["text"],
    }
    | _prompt
    | _model
    | StrOutputParser()
)

# PII Detection logic
analyzer = AnalyzerEngine()


# You can customize this to detect any PII
def _detect_pii(inputs: dict) -> bool:
    analyzer_results = analyzer.analyze(text=inputs["text"], language="en")
    return bool(analyzer_results)


# Add logic to route on whether PII has been detected
def _route_on_pii(inputs: dict):
    if inputs["pii_detected"]:
        # Response if PII is detected
        return "Sorry, I can't answer questions that involve PII"
    else:
        return chat_chain


# Final chain
chain = RunnablePassthrough.assign(
    # First detect PII
    pii_detected=_detect_pii
) | {
    # Then use this information to generate the response
    "response": _route_on_pii,
    # Return boolean of whether PII is detected so client can decided
    # whether or not to include in chat history
    "pii_detected": lambda x: x["pii_detected"],
}


# Add typing for playground
class ChainInput(BaseModel):
    text: str
    chat_history: List[Tuple[str, str]]


chain = chain.with_types(input_type=ChainInput)
