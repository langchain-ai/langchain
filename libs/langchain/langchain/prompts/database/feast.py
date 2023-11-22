"""
A prompt template that automates retrieving rows from Feast and making their
content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Dict, List, Tuple, Union

if typing.TYPE_CHECKING:
    from feast.entity import Entity
    from feast.feature_store import FeatureStore
    from feast.feature_view import FeatureView

from langchain.prompts.database.converter_prompt_template import ConverterPromptTemplate
from langchain.pydantic_v1 import root_validator

FeatureRetrievalPrescriptionType = Union[
    Tuple[str, str], Tuple[str, str, bool], Tuple[str, str, bool, Any]
]
FieldMapperType = Dict[str, FeatureRetrievalPrescriptionType]

DEFAULT_ADMIT_NULLS = True


def _feast_get_entity_by_name(store: FeatureStore, entity_name: str) -> Entity:
    return [ent for ent in store.list_entities() if ent.name == entity_name][0]


def _feast_get_entity_join_keys(entity: Entity) -> List[str]:
    if hasattr(entity, "join_keys"):
        # Feast plans to replace `join_key: str` with `join_keys: List[str]`
        return list(entity.join_keys)
    else:
        return [entity.join_key]


def _feast_get_feature_view_by_name(
    store: FeatureStore, feature_view_name: str
) -> FeatureView:
    return [
        fview for fview in store.list_feature_views() if fview.name == feature_view_name
    ][0]


def _ensure_full_extraction_tuple(
    tpl: Tuple[Any, ...], admit_nulls: bool
) -> Tuple[Any, ...]:
    if len(tpl) < 2:
        raise ValueError(
            "At least feature_view and feature_name are required in the field_mapper."
        )
    elif len(tpl) == 2:
        return tuple(list(tpl) + [admit_nulls, None])
    elif len(tpl) == 3:
        return tuple(list(tpl) + [None])
    elif len(tpl) == 4:
        return tpl
    else:
        raise ValueError(
            "Cannot specify more than (feature_view, feature_name, "
            "admit_nulls, default) in the field_mapper"
        )


class FeastReaderPromptTemplate(ConverterPromptTemplate):
    feature_store: Any  # FeatureStore

    field_mapper: FieldMapperType

    admit_nulls: bool = DEFAULT_ADMIT_NULLS

    @root_validator(pre=True)
    def check_and_provide_converter(cls, values: Dict) -> Dict:
        converter_info = cls._prepare_reader_info(
            feature_store=values["feature_store"],
            field_mapper=values["field_mapper"],
            admit_nulls=values.get("admit_nulls", DEFAULT_ADMIT_NULLS),
        )
        for k, v in converter_info.items():
            values[k] = v
        return values

    @staticmethod
    def _prepare_reader_info(
        feature_store: FeatureStore,
        field_mapper: FieldMapperType,
        admit_nulls: bool,
    ) -> Dict[str, Any]:
        try:
            pass
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import feast python package. "
                'Please install it with `pip install "feast>=0.26"`.'
            )
        # normalize the field mapper to a 4-tuple
        #   (f_view_name, f_name, admit_nulls, default)
        norm_field_mapper = {
            k: _ensure_full_extraction_tuple(v, admit_nulls)
            for k, v in field_mapper.items()
        }
        # inspection of the store to build the getter and the var names:
        required_f_views = [
            _feast_get_feature_view_by_name(feature_store, f_view_name)
            for (f_view_name, _, _, _) in norm_field_mapper.values()
        ]
        required_entity_names = {
            ent for f_view in required_f_views for ent in f_view.entities
        }
        join_keys = sorted(
            {
                join_key
                for entity_name in required_entity_names
                for join_key in _feast_get_entity_join_keys(
                    _feast_get_entity_by_name(feature_store, entity_name)
                )
            }
        )

        def _converter(args_dict: Dict[str, Any]) -> Dict[str, Any]:
            feature_vector = feature_store.get_online_features(
                features=[
                    f"{fview}:{fname}"
                    for _, (fview, fname, _, _) in norm_field_mapper.items()
                ],
                entity_rows=[args_dict],
            ).to_dict()
            #
            retrieved_variables = {
                vname: _check_for_null_f_val(
                    f"{fview}:{fname}",
                    feature_vector[fname][0],
                    ok_nulls,
                    default_f_value,
                )
                for vname, (
                    fview,
                    fname,
                    ok_nulls,
                    default_f_value,
                ) in norm_field_mapper.items()
            }
            #
            return retrieved_variables

        return {
            "converter": _converter,
            "converter_output_variables": list(norm_field_mapper.keys()),
            "converter_input_variables": join_keys,
        }

    @property
    def _prompt_type(self) -> str:
        return "feast-reader-prompt-template"


def _check_for_null_f_val(
    f_full_name: str, f_raw_value: Any, ok_nulls: bool, default_f_value: Any
) -> Any:
    if f_raw_value is not None:
        return f_raw_value
    else:
        if ok_nulls:
            return default_f_value
        else:
            raise ValueError('Null feature value found for "%s"' % f_full_name)
