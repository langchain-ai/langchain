from langchain.automaton.prompt_generation import AdapterBasedTranslator
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain.automaton.typedefs import FunctionCallResponse


def test_adapter_based_generator() -> None:
    translator = AdapterBasedTranslator()

    assert translator.to_messages(
        [
            SystemMessage(content="System"),
            AIMessage(content="Hi"),
            HumanMessage(content="Hello"),
        ]
    ) == [
        SystemMessage(content="System"),
        AIMessage(content="Hi"),
        HumanMessage(content="Hello"),
    ]

    translator = AdapterBasedTranslator(
        msg_adapters={
            HumanMessage: lambda m: AIMessage(content=m.content),
            FunctionCallResponse: lambda m: HumanMessage(
                content=f"Observation: {m.result}"
            ),
        }
    )

    assert translator.to_messages(
        [
            SystemMessage(content="System"),
            AIMessage(content="Hi"),
            HumanMessage(content="Hello"),
            FunctionCallResponse(name="func", result="result"),
        ]
    ) == [
        SystemMessage(content="System"),
        AIMessage(content="Hi"),
        AIMessage(content="Hello"),
        HumanMessage(content="Observation: result"),
    ]
