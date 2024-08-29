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
    :exclude-members: construct, copy, dict, from_orm, parse_file, parse_obj, parse_raw, schema, schema_json, update_forward_refs, validate, json, is_lc_serializable, to_json, to_json_not_implemented, lc_secrets, lc_attributes, lc_id, get_lc_namespace


    {% block attributes %}
    {% endblock %}

.. example_links:: {{ objname }}
