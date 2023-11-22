"""
A prompt template that automates retrieving rows from multiple tables in
Cassandra and making their content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session

from langchain.prompts.database.convertor_prompt_template import ConvertorPromptTemplate
from langchain.pydantic_v1 import root_validator

RowToValueType = Union[str, Callable[[Any], Any]]
FieldMapperType = Dict[
    str,
    Union[
        Tuple[str, RowToValueType],
        Tuple[str, RowToValueType, bool],
        Tuple[str, RowToValueType, bool, Any],
    ],
]

DEFAULT_ADMIT_NULLS = True


class CassandraReaderPromptTemplate(ConvertorPromptTemplate):
    session: Optional[Any] = None  # Session

    keyspace: Optional[str] = None

    field_mapper: FieldMapperType

    input_variables: List[str] = []

    admit_nulls: bool = DEFAULT_ADMIT_NULLS

    @root_validator(pre=True)
    def check_and_provide_convertor(cls, values: Dict) -> Dict:
        convertor_info = cls._prepare_reader_info(
            session=values.get("session"),
            keyspace=values.get("keyspace"),
            field_mapper=values["field_mapper"],
            admit_nulls=values.get("admit_nulls", DEFAULT_ADMIT_NULLS),
        )
        for k, v in convertor_info.items():
            values[k] = v
        # values["input_variables"] = values.get("input_variables", [])
        return values

    @staticmethod
    def _prepare_reader_info(
        session: Session,
        keyspace: str,
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
        _convertor = MultiTableCassandraReader(
            session=session,
            keyspace=keyspace,
            field_mapper=field_mapper,
            admit_nulls=admit_nulls,
        )

        return {
            "convertor": _convertor.dictionary_based_call,
            "convertor_output_variables": _convertor.output_parameters,
            "convertor_input_variables": _convertor.input_parameters,
        }

    @property
    def _prompt_type(self) -> str:
        return "cassandra-reader-prompt-template"
