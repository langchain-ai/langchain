from tests.unit_tests.migrate.integration.case import Case
from tests.unit_tests.migrate.integration.cases import imports
from tests.unit_tests.migrate.integration.file import File
from tests.unit_tests.migrate.integration.folder import Folder

cases = [
    Case(
        name="empty",
        source=File("__init__.py", content=[]),
        expected=File("__init__.py", content=[]),
    ),
    *imports.cases,
]
before = Folder("project", *[case.source for case in cases])
expected = Folder("project", *[case.expected for case in cases])
