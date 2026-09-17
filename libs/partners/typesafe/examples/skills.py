"""Select agent skills with TypeSafe.

Run against the live API:

    export TYPESAFE_API_KEY=...
    uv run python examples/skills.py
"""

from langchain_core.messages import HumanMessage

from langchain_typesafe import SkillsMiddleware

CODE_REVIEW = """---
name: code-review
description: Review a code diff for correctness and style problems.
---

Report findings most severe first.
"""

RECIPES = """---
name: recipe-writing
description: Write cooking recipes and suggest ingredient substitutions.
---

Give quantities in both metric and imperial units.
"""

REQUESTS = [
    "Take a look at this diff and tell me what's wrong.",
    "What can I use instead of buttermilk?",
    "What time zone is Lisbon in?",
]


def main() -> None:
    """Score both skills against each request and print what was selected."""
    middleware = SkillsMiddleware(skills=[CODE_REVIEW, RECIPES])

    for request in REQUESTS:
        state = {"messages": [HumanMessage(request)]}
        selected = middleware.before_agent(state, None)["selected_skills"]  # type: ignore[arg-type]
        print(f"{selected or '[]'} <- {request}")


if __name__ == "__main__":
    main()
