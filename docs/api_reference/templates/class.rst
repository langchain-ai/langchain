{{ objname }}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ item }}
   {%- endfor %}

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}


.. example_links:: {{ objname }}
