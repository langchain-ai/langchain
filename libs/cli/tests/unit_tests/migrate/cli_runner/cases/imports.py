from tests.unit_tests.migrate.cli_runner.case import Case
from tests.unit_tests.migrate.cli_runner.file import File

cases = [
    Case(
        name="Imports",
        source=File(
            "app.py",
            content=[
                "from langchain_community.chat_models import ChatOpenAI",
                "",
                "",
                "class foo:",
                "    a: int",
                "",
                "chain = ChatOpenAI()",
            ],
        ),
        expected=File(
            "app.py",
            content=[
                "from langchain_openai import ChatOpenAI",
                "",
                "",
                "class foo:",
                "    a: int",
                "",
                "chain = ChatOpenAI()",
            ],
        ),
    ),
]
