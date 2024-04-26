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
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypeVar

import libcst as cst
import libcst.matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor

HERE = os.path.dirname(__file__)


def _load_migrations_by_file(path: str):
    migrations_path = os.path.join(HERE, path)
    with open(migrations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


T = TypeVar("T")


def _deduplicate_in_order(
    seq: Iterable[T], key: Callable[[T], str] = lambda x: x
) -> List[T]:
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (key(x) in seen or seen_add(key(x)))]


def _load_migrations():
    """Load the migrations from the JSON file."""
    # Later earlier ones have higher precedence.
    paths = [
        "migrations_v0.2_partner.json",
        "migrations_v0.2.json",
    ]

    data = []
    for path in paths:
        data.extend(_load_migrations_by_file(path))

    data = _deduplicate_in_order(data, key=lambda x: x[0])

    imports: Dict[str, Tuple[str, str]] = {}

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


IMPORTS = _load_migrations()


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


IMPORT_INFOS = [
    ImportInfo(
        import_from=get_import_from_from_str(import_str),
        import_str=import_str,
        to_import_str=to_import_str,
    )
    for import_str, to_import_str in IMPORTS.items()
]
IMPORT_MATCH = m.OneOf(*[info.import_from for info in IMPORT_INFOS])


class ReplaceImportsCodemod(VisitorBasedCodemodCommand):
    @m.leave(IMPORT_MATCH)
    def leave_replace_import(
        self, _: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        for import_info in IMPORT_INFOS:
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
                    names[-1] = names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
                    updated_node = updated_node.with_changes(names=names)
                else:
                    return cst.RemoveFromParent()  # type: ignore[return-value]
        return updated_node


if __name__ == "__main__":
    import textwrap

    from rich.console import Console

    console = Console()

    source = textwrap.dedent(
        """
        from pydantic.settings import BaseSettings
        from pydantic.color import Color
        from pydantic.payment import PaymentCardNumber, PaymentCardBrand
        from pydantic import Color
        from pydantic import Color as Potato


        class Potato(BaseSettings):
            color: Color
            payment: PaymentCardNumber
            brand: PaymentCardBrand
            potato: Potato
        """
    )
    console.print(source)
    console.print("=" * 80)

    mod = cst.parse_module(source)
    context = CodemodContext(filename="main.py")
    wrapper = cst.MetadataWrapper(mod)
    command = ReplaceImportsCodemod(context=context)

    mod = wrapper.visit(command)
    wrapper = cst.MetadataWrapper(mod)
    command = AddImportsVisitor(context=context)  # type: ignore[assignment]
    mod = wrapper.visit(command)
    console.print(mod.code)
