import sqlalchemy

import langchain  # noqa: F401


def test_configure_mappers() -> None:
    sqlalchemy.orm.configure_mappers()
