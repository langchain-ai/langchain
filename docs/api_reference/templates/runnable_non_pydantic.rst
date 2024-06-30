:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. NOTE:: {{objname}} implements the standard :py:class:`Runnable Interface <langchain_core.runnables.base.Runnable>`. 🏃

    The :py:class:`Runnable Interface <langchain_core.runnables.base.Runnable>` has additional methods that are available on runnables, such as :py:meth:`with_types <langchain_core.runnables.base.Runnable.with_types>`, :py:meth:`with_retry <langchain_core.runnables.base.Runnable.with_retry>`, :py:meth:`assign <langchain_core.runnables.base.Runnable.assign>`, :py:meth:`bind <langchain_core.runnables.base.Runnable.bind>`, :py:meth:`get_graph <langchain_core.runnables.base.Runnable.get_graph>`, and more.

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% for item in methods %}
   .. automethod:: {{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}


.. example_links:: {{ objname }}
