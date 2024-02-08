:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block attributes %}
    {% for item in attributes %}
    .. autoattribute:: {{ item }}
    {% endfor %}
    {% endblock %}

.. example_links:: {{ objname }}
