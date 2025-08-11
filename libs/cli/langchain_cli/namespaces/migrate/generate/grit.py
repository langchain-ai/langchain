"""Migration as Grit file."""


def split_package(package: str) -> tuple[str, str]:
    """Split a package name into the containing package and the final name."""
    parts = package.split(".")
    return ".".join(parts[:-1]), parts[-1]


def dump_migrations_as_grit(name: str, migration_pairs: list[tuple[str, str]]) -> str:
    """Dump the migration pairs as a Grit file."""
    remapped = ",\n".join(
        [
            f"""
            [
                `{split_package(from_module)[0]}`,
                `{split_package(from_module)[1]}`,
                `{split_package(to_module)[0]}`,
                `{split_package(to_module)[1]}`
            ]
            """
            for from_module, to_module in migration_pairs
        ],
    )
    pattern_name = f"langchain_migrate_{name}"
    return f"""
language python

// This migration is generated automatically - do not manually edit this file
pattern {pattern_name}() {{
  find_replace_imports(list=[
{remapped}
  ])
}}

// Add this for invoking directly
{pattern_name}()
"""
