from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

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
_model = ChatOpenAI()

chat_chain = {
    "chat_history": lambda x: _format_chat_history(x["chat_history"]),
    "text": lambda x: x["text"],
} | _prompt | _model | StrOutputParser()


# You can customize this to detect any PII
def _detect_pii(inputs: dict) -> bool:
    analyzer_results = analyzer.analyze(text=inputs["text"], language="en")
    return bool(analyzer_results)


def _route_on_pii(inputs: dict):
    if inputs["pii_detected"]:
        return "Sorry, I can't answer questions that involve PII"
    else:
        return chat_chain

class ChainInput(BaseModel):
    text: str
    chat_history: List[Tuple[str, str]]


chain = RunnablePassthrough.assign(
    pii_detected=_detect_pii
).with_types(input_type=ChainInput) | {
    "response": _route_on_pii,
    "pii_detected": lambda x: x["pii_detected"]
}
