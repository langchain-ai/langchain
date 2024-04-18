"""Test ChatAnthropic chat model."""

from enum import Enum
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_anthropic.experimental import ChatAnthropicTools

MODEL_NAME = "claude-3-sonnet-20240229"
BIG_MODEL_NAME = "claude-3-opus-20240229"

#####################################
### Test Basic features, no tools ###
#####################################


def test_stream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatAnthropicTools."""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatAnthropicTools."""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatAnthropicTools."""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatAnthropicTools."""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatAnthropicTools."""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_system_invoke() -> None:
    """Test invoke tokens with a system message"""
    llm = ChatAnthropicTools(model_name=MODEL_NAME)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert cartographer. If asked, you are a cartographer. "
                "STAY IN CHARACTER",
            ),
            ("human", "Are you a mathematician?"),
        ]
    )

    chain = prompt | llm

    result = chain.invoke({})
    assert isinstance(result.content, str)


##################
### Test Tools ###
##################


def test_with_structured_output() -> None:
    class Person(BaseModel):
        name: str
        age: int

    chain = ChatAnthropicTools(
        model_name=BIG_MODEL_NAME,
        temperature=0,
        default_headers={"anthropic-beta": "tools-2024-04-04"},
    ).with_structured_output(Person)
    result = chain.invoke("Erick is 27 years old")
    assert isinstance(result, Person)
    assert result.name == "Erick"
    assert result.age == 27


def test_anthropic_complex_structured_output() -> None:
    class ToneEnum(str, Enum):
        positive = "positive"
        negative = "negative"

    class Email(BaseModel):
        """Relevant information about an email."""

        sender: Optional[str] = Field(
            None, description="The sender's name, if available"
        )
        sender_phone_number: Optional[str] = Field(
            None, description="The sender's phone number, if available"
        )
        sender_address: Optional[str] = Field(
            None, description="The sender's address, if available"
        )
        action_items: List[str] = Field(
            ..., description="A list of action items requested by the email"
        )
        topic: str = Field(
            ..., description="High level description of what the email is about"
        )
        tone: ToneEnum = Field(..., description="The tone of the email.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "What can you tell me about the following email? Make sure to answer in the correct format: {email}",  # noqa: E501
            ),
        ]
    )

    llm = ChatAnthropicTools(
        temperature=0,
        model_name=BIG_MODEL_NAME,
        default_headers={"anthropic-beta": "tools-2024-04-04"},
    )

    extraction_chain = prompt | llm.with_structured_output(Email)

    response = extraction_chain.invoke(
        {
            "email": "From: Erick. The email is about the new project. The tone is positive. The action items are to send the report and to schedule a meeting."  # noqa: E501
        }
    )  # noqa: E501
    assert isinstance(response, Email)
