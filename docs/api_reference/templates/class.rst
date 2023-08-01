:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

{% if '_value2member_map_' in  all_attributes %}
    {% set classType = "enum" %}
{% else %}
    {% set classType = "default" %}
{% endif %}

.. autoclass:: {{ objname }}

   {% if classType == "enum" %}
       {% if attributes %}
       .. rubric:: {{ _('Attributes') }}
       {% endif %}
   {% else %}
       {% if attributes %}
       .. rubric:: {{ _('Attributes') }}
       {% endif %}

       {% block methods %}
       {% if methods %}
       .. rubric:: {{ _('Methods') }}

       .. autosummary::
       {% for item in methods %}
          ~{{ name }}.{{ item }}
       {%- endfor %}
       {% endif %}
       {% endblock %}

       {% block attributes %}
       {% if attributes %}
       .. rubric:: {{ _('Attributes') }}

       .. autosummary::
       {% for item in attributes %}
          ~{{ name }}.{{ item }}
       {%- endfor %}
       {% endif %}
       {% endblock %}
    {% endif %}

.. example_links:: {{ objname }}
