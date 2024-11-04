"""Handle a test case where the import is updated and may involve an alias change."""

from tests.unit_tests.migrate.cli_runner.case import Case
from tests.unit_tests.migrate.cli_runner.file import File

# The test case right now make sure that if we update the import
# of RunnableMap to RunnableParallel then the code that's using RunnableMap
# should be updated as well (or else we keep importing RunnableMap.)
cases = [
    Case(
        name="Imports",
        source=File(
            "app.py",
            content=[
                "from langchain.runnables import RunnableMap",
                "",
                "chain = RunnableMap({})",
            ],
        ),
        expected=File(
            "app.py",
            content=[
                "from langchain_core.runnables import RunnableMap",
                "",
                "chain = RunnableMap({})",
            ],
        ),
    ),
]
""
