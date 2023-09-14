:mod:`{{module}}`.{{objname}}
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

    {% block attributes %}
    {% endblock %}

.. example_links:: {{ objname }}
