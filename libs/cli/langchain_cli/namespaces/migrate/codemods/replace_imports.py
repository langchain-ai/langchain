"""
# Adapted from bump-pydantic
# https://github.com/pydantic/bump-pydantic

This codemod deals with the following cases:

1. `from pydantic import BaseSettings`
2. `from pydantic.settings import BaseSettings`
3. `from pydantic import BaseSettings as <name>`
4. `from pydantic.settings import BaseSettings as <name>`  # TODO: This is not working.
5. `import pydantic` -> `pydantic.BaseSettings`
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Type, TypeVar

import libcst as cst
import libcst.matchers as m
from libcst.codemod import VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor

HERE = os.path.dirname(__file__)


def _load_migrations_by_file(path: str):
    migrations_path = os.path.join(HERE, "migrations", path)
    with open(migrations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # new migrations
    new_migrations = []
    for migration in data:
        old = migration[0].split(".")[-1]
        new = migration[1].split(".")[-1]

        if old == new:
            new_migrations.append(migration)

    return new_migrations


T = TypeVar("T")


def _deduplicate_in_order(
    seq: Iterable[T], key: Callable[[T], str] = lambda x: x
) -> List[T]:
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (key(x) in seen or seen_add(key(x)))]


def _load_migrations_from_fixtures(paths: List[str]) -> List[Tuple[str, str]]:
    """Load migrations from fixtures."""
    data = []
    for path in paths:
        data.extend(_load_migrations_by_file(path))
    data = _deduplicate_in_order(data, key=lambda x: x[0])
    return data


def _load_migrations(paths: List[str]):
    """Load the migrations from the JSON file."""
    # Later earlier ones have higher precedence.
    imports: Dict[str, Tuple[str, str]] = {}
    data = _load_migrations_from_fixtures(paths)

    for old_path, new_path in data:
        # Parse the old parse which is of the format 'langchain.chat_models.ChatOpenAI'
        # into the module and class name.
        old_parts = old_path.split(".")
        old_module = ".".join(old_parts[:-1])
        old_class = old_parts[-1]
        old_path_str = f"{old_module}:{old_class}"

        # Parse the new parse which is of the format 'langchain.chat_models.ChatOpenAI'
        # Into a 2-tuple of the module and class name.
        new_parts = new_path.split(".")
        new_module = ".".join(new_parts[:-1])
        new_class = new_parts[-1]
        new_path_str = (new_module, new_class)

        imports[old_path_str] = new_path_str

    return imports


def resolve_module_parts(module_parts: list[str]) -> m.Attribute | m.Name:
    """Converts a list of module parts to a `Name` or `Attribute` node."""
    if len(module_parts) == 1:
        return m.Name(module_parts[0])
    if len(module_parts) == 2:
        first, last = module_parts
        return m.Attribute(value=m.Name(first), attr=m.Name(last))
    last_name = module_parts.pop()
    attr = resolve_module_parts(module_parts)
    return m.Attribute(value=attr, attr=m.Name(last_name))


def get_import_from_from_str(import_str: str) -> m.ImportFrom:
    """Converts a string like `pydantic:BaseSettings` to     Examples:
    >>> get_import_from_from_str("pydantic:BaseSettings")
    ImportFrom(
        module=Name("pydantic"),
        names=[ImportAlias(name=Name("BaseSettings"))],
    )
    >>> get_import_from_from_str("pydantic.settings:BaseSettings")
    ImportFrom(
        module=Attribute(value=Name("pydantic"), attr=Name("settings")),
        names=[ImportAlias(name=Name("BaseSettings"))],
    )
    >>> get_import_from_from_str("a.b.c:d")
    ImportFrom(
        module=Attribute(
            value=Attribute(value=Name("a"), attr=Name("b")), attr=Name("c")
        ),
        names=[ImportAlias(name=Name("d"))],
    )
    """
    module, name = import_str.split(":")
    module_parts = module.split(".")
    module_node = resolve_module_parts(module_parts)
    return m.ImportFrom(
        module=module_node,
        names=[m.ZeroOrMore(), m.ImportAlias(name=m.Name(value=name)), m.ZeroOrMore()],
    )


@dataclass
class ImportInfo:
    import_from: m.ImportFrom
    import_str: str
    to_import_str: tuple[str, str]


RULE_TO_PATHS = {
    "langchain_to_community": ["langchain_to_community.json"],
    "langchain_to_core": ["langchain_to_core.json"],
    "community_to_core": ["community_to_core.json"],
    "langchain_to_text_splitters": ["langchain_to_text_splitters.json"],
    "community_to_partner": [
        "anthropic.json",
        "fireworks.json",
        "ibm.json",
        "openai.json",
        "pinecone.json",
        "astradb.json",
    ],
}


def generate_import_replacer(rules: List[str]) -> Type[VisitorBasedCodemodCommand]:
    """Generate a codemod to replace imports."""
    paths = []
    for rule in rules:
        if rule not in RULE_TO_PATHS:
            raise ValueError(f"Unknown rule: {rule}. Use one of {RULE_TO_PATHS.keys()}")

        paths.extend(RULE_TO_PATHS[rule])

    imports = _load_migrations(paths)

    import_infos = [
        ImportInfo(
            import_from=get_import_from_from_str(import_str),
            import_str=import_str,
            to_import_str=to_import_str,
        )
        for import_str, to_import_str in imports.items()
    ]
    import_match = m.OneOf(*[info.import_from for info in import_infos])

    class ReplaceImportsCodemod(VisitorBasedCodemodCommand):
        @m.leave(import_match)
        def leave_replace_import(
            self, _: cst.ImportFrom, updated_node: cst.ImportFrom
        ) -> cst.ImportFrom:
            for import_info in import_infos:
                if m.matches(updated_node, import_info.import_from):
                    aliases: Sequence[cst.ImportAlias] = updated_node.names  # type: ignore
                    # If multiple objects are imported in a single import statement,
                    # we need to remove only the one we're replacing.
                    AddImportsVisitor.add_needed_import(
                        self.context, *import_info.to_import_str
                    )
                    if len(updated_node.names) > 1:  # type: ignore
                        names = [
                            alias
                            for alias in aliases
                            if alias.name.value != import_info.to_import_str[-1]
                        ]
                        names[-1] = names[-1].with_changes(
                            comma=cst.MaybeSentinel.DEFAULT
                        )
                        updated_node = updated_node.with_changes(names=names)
                    else:
                        return cst.RemoveFromParent()  # type: ignore[return-value]
            return updated_node

    return ReplaceImportsCodemod
