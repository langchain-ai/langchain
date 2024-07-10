"""Verify that the code migrations do not involve alias changes.

Migration script only updates imports not the rest of the code that uses the
import.
"""

from langchain_cli.namespaces.migrate.codemods.replace_imports import (
    RULE_TO_PATHS,
    _load_migrations_from_fixtures,
)


def test_migration_files() -> None:
    """Generate a codemod to replace imports."""
    errors = []

    for paths in list(RULE_TO_PATHS.values()):
        for path in paths:
            migrations = _load_migrations_from_fixtures([path])

            for migration in migrations:
                old = migration[0].split(".")[-1]
                new = migration[1].split(".")[-1]
                if old != new:
                    errors.append((path, migration))

    if errors:
        raise ValueError(
            f"Migration involves an alias change: {errors}. The "
            f"migration script does not currently support "
            f"corresponding code changes."
        )
