"""
A prompt template that automates retrieving rows from multiple tables in
Cassandra and making their content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

if typing.TYPE_CHECKING:
    pass

from langchain_core.pydantic_v1 import root_validator

from langchain_community.prompts.converter_prompt_template import (
    ConverterPromptTemplate,
)

RowToValueType = Union[str, Callable[[Any], Any]]
FieldMapperType = Dict[
    str,
    Tuple[Any, ...],
]
# more specifically, as a guide to usage:
# FieldMapperType = Dict[
#     str,
#     Union[
#         Tuple[str, RowToValueType],
#         Tuple[str, RowToValueType, bool],
#         Tuple[str, RowToValueType, bool, Any],
#     ],
# ]


DEFAULT_ADMIT_NULLS = True


class CassandraReaderPromptTemplate(ConverterPromptTemplate):
    """
    A prompt that is bound to a Cassandra keyspace in such a way that the inputs
    to the format() method are primary key(s) used for row(s) lookup,
    subsequently used to fill the actual template with other columns
    in the tables' rows.


    Example:
        .. code-block:: python

            from langchain_community.prompts import CassandraReaderPromptTemplate

            import cassio
            cassio.init(auto=True)  # Requires env. variables, see CassIO docs

            prompt_fstring = (
              "Please write a {adj} greeting for {user_name} "
              "(you may use the informal name '{short_name}' where appropriate)"
            )

            my_template = CassandraReaderPromptTemplate(
              template=prompt_fstring,
              field_mapper={
                "user_name": ("demo_users", "user_name"),
                "short_name": ("demo_users", "short_name"),
              },
            )

            # table 'demo_users' has 'user_id' as primary key, so:
            prompt = my_template.format(user_id="john_doe", adj="lackadaisical")

    Args:
        session: an open Cassandra session.
            Leave unspecified to use the global cassio init (see below)
        keyspace: the keyspace to use for storing the cache.
            Leave unspecified to use the global cassio init (see below)
        template: a template string to format the prompt (f-string by default)
        field_mapper: a map from variables in the template to one of:
            (table_name, column_name)
            (table_name, column_name, admit_nulls)
            (table_name, column_name, admit_nulls, default_if_null_found)
            (table_name, row_function)
            (table_name, row_function, admit_nulls)
            (table_name, row_function, admit_nulls, default_if_null_found)
            Above, 'row_function' is a function of a Cassandra 'row' returning
            a value (for row-wide custom value extraction).
        input_variables: provide additional template variables found in the
            template, other than those specified in the field_mapper.
        admit_nulls: whether to raise an error or proceed when the required
        row is not found, or a None is found at the required column. This is
        an extraction-wide setting, overridable per-variable through the third
        (and fourth) items in the field_mapper tuple.

    Note:
        The session and keyspace parameters, when left out (or passed as None),
        fall back to the globally-available cassio settings if are available.
        In other words, if a previously-run 'cassio.init(...)' has been
        executed previously anywhere in the code, Cassandra-based objects
        need not specify the connection parameters at all.
    """

    session: Optional[Any] = None  # Session

    keyspace: Optional[str] = None

    field_mapper: FieldMapperType

    input_variables: List[str] = []

    admit_nulls: bool = DEFAULT_ADMIT_NULLS

    @root_validator(pre=True)
    def check_and_provide_converter(cls, values: Dict) -> Dict:
        converter_info = cls._prepare_reader_info(
            session=values.get("session"),
            keyspace=values.get("keyspace"),
            field_mapper=values["field_mapper"],
            admit_nulls=values.get("admit_nulls", DEFAULT_ADMIT_NULLS),
        )
        for k, v in converter_info.items():
            values[k] = v
        return values

    @staticmethod
    def _prepare_reader_info(
        session: Optional[Any],  # Session
        keyspace: Optional[str],
        field_mapper: FieldMapperType,
        admit_nulls: bool,
    ) -> Dict[str, Any]:
        try:
            from cassio.db_reader import MultiTableCassandraReader
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        #
        _converter = MultiTableCassandraReader(
            session=session,
            keyspace=keyspace,
            field_mapper=field_mapper,
            admit_nulls=admit_nulls,
        )

        return {
            "converter": _converter.dictionary_based_call,
            "converter_output_variables": _converter.output_parameters,
            "converter_input_variables": _converter.input_parameters,
        }

    @property
    def _prompt_type(self) -> str:
        return "cassandra-reader-prompt-template"
