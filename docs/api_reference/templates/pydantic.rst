:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autopydantic_model:: {{ objname }}
    :model-show-json: False
    :model-show-config-summary: False
    :model-show-validator-members: False
    :model-show-field-summary: False
    :model-signature-prefix: class
    :field-signature-prefix: param
    :special-members: __init__
    :undoc-members:
    :member-order: groupwise
    :show-inheritance: True

    {% block attributes %}
    {% endblock %}

.. example_links:: {{ objname }}
