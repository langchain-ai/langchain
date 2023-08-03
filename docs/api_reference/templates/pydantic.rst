:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autopydantic_model:: {{ objname }}
    :model-show-json: False
    :model-show-config-summary: False
    :model-show-validator-members: False
    :model-show-field-summary: False
    :field-signature-prefix: param
    :undoc-members:
    :member-order: groupwise
    :show-inheritance: True

    {% block attributes %}
    {% endblock %}

.. example_links:: {{ objname }}
