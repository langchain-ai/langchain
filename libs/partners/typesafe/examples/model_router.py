"""Route an agent's model with TypeSafe.

Run against the live API:

    export TYPESAFE_API_KEY=... OPENAI_API_KEY=...
    uv run python examples/model_router.py
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_typesafe import ModelChoice, ModelRouterMiddleware

REQUESTS = [
    "What is the capital of France?",
    "Our checkout service deadlocks under load about once a day. Help me find why.",
]


def main() -> None:
    """Classify two requests and print the model each one routes to."""
    router = ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(
                model=ChatOpenAI(model="gpt-5-nano"),
                criteria="Trivial lookups and one-line edits.",
            ),
            "powerful": ModelChoice(
                model=ChatOpenAI(model="gpt-5"),
                criteria="Multi-step reasoning, architecture, and debugging.",
            ),
        },
        instructions="Choose the least costly model that can do the task well.",
        default_route="powerful",
    )

    for request in REQUESTS:
        state = {"messages": [HumanMessage(request)]}
        route = router.before_agent(state, None)["model_route"]  # type: ignore[arg-type]
        print(f"{route:<9} <- {request}")


if __name__ == "__main__":
    main()
