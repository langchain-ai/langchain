from enum import Enum
from typing import List, Type

from libcst.codemod import ContextAwareTransformer
from libcst.codemod.visitors import AddImportsVisitor, RemoveImportsVisitor

from langchain_cli.namespaces.migrate.codemods.replace_imports import (
    ReplaceImportsCodemod,
)


class Rule(str, Enum):
    R001 = "R001"
    """Replace imports that have been moved."""


def gather_codemods(disabled: List[Rule]) -> List[Type[ContextAwareTransformer]]:
    codemods: List[Type[ContextAwareTransformer]] = []

    if Rule.R001 not in disabled:
        codemods.append(ReplaceImportsCodemod)

    # Those codemods need to be the last ones.
    codemods.extend([RemoveImportsVisitor, AddImportsVisitor])
    return codemods
