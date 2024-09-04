from langchain_community.chains.sineps.intent_router import (
    Route,
    SinepsIntentRouterChain,
)


def test_sineps_intent_router() -> None:
    routes = [
        Route(
            key="greet",
            name="Greet",
            description="Greet the user",
            utterances=["hello", "hi"],
        ),
        Route(
            key="goodbye",
            name="Goodbye",
            description="Say goodbye to the user",
            utterances=["goodbye", "bye"],
        ),
    ]

    chain = SinepsIntentRouterChain(routes=routes, allow_none=True)
    output = chain.invoke({"query": "hello"})
    assert output["key"] == "greet"
