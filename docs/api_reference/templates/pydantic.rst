{{ objname }}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autopydantic_model:: {{ objname }}
    :model-show-json: False
    :model-show-config-summary: False
    :model-show-validator-members: False
    :model-show-field-summary: False
    :field-signature-prefix: param
    :members:
    :undoc-members:
    :inherited-members:
    :member-order: groupwise
    :show-inheritance: True
    :special-members: __call__
    :exclude-members: construct, copy, dict, from_orm, parse_file, parse_obj, parse_raw, schema, schema_json, update_forward_refs, validate, json, is_lc_serializable, to_json, to_json_not_implemented, lc_secrets, lc_attributes, lc_id, get_lc_namespace, model_construct, model_copy, model_dump, model_dump_json, model_parametrized_name, model_post_init, model_rebuild, model_validate, model_validate_json, model_validate_strings, model_extra, model_fields_set, model_json_schema


    {% block attributes %}
    {% endblock %}

.. example_links:: {{ objname }}
