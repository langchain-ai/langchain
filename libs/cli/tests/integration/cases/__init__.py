from tests.integration.case import Case
from tests.integration.cases import imports
from tests.integration.file import File
from tests.integration.folder import Folder

cases = [
    Case(
        name="empty",
        source=File("__init__.py", content=[]),
        expected=File("__init__.py", content=[]),
    ),
    *imports,
]
before = Folder("project", *[case.source for case in cases])
expected = Folder("project", *[case.expected for case in cases])
