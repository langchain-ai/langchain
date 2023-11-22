"""
A prompt template that automates retrieving rows from multiple tables in
Cassandra and making their content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

if typing.TYPE_CHECKING:
    pass

from langchain.prompts.database.converter_prompt_template import ConverterPromptTemplate
from langchain.pydantic_v1 import root_validator

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
