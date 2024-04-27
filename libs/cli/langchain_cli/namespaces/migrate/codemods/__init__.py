from enum import Enum
from typing import List, Type

from libcst.codemod import ContextAwareTransformer
from libcst.codemod.visitors import AddImportsVisitor, RemoveImportsVisitor

from langchain_cli.namespaces.migrate.codemods.replace_imports import (
    generate_import_replacer,
)


class Rule(str, Enum):
    langchain_to_community = "langchain_to_community"
    """Replace deprecated langchain imports with current ones in community."""
    langchain_to_core = "langchain_to_core"
    """Replace deprecated langchain imports with current ones in core."""
    langchain_to_text_splitters = "langchain_to_text_splitters"
    """Replace deprecated langchain imports with current ones in text splitters."""
    community_to_core = "community_to_core"
    """Replace deprecated community imports with current ones in core."""
    community_to_partner = "community_to_partner"
    """Replace deprecated community imports with current ones in partner."""


def gather_codemods(disabled: List[Rule]) -> List[Type[ContextAwareTransformer]]:
    """Gather codemods based on the disabled rules."""
    codemods: List[Type[ContextAwareTransformer]] = []

    # Import rules
    import_rules = {
        Rule.langchain_to_community,
        Rule.langchain_to_core,
        Rule.community_to_core,
        Rule.community_to_partner,
        Rule.langchain_to_text_splitters,
    }

    # Find active import rules
    active_import_rules = import_rules - set(disabled)

    if active_import_rules:
        codemods.append(generate_import_replacer(active_import_rules))
    # Those codemods need to be the last ones.
    codemods.extend([RemoveImportsVisitor, AddImportsVisitor])
    return codemods
