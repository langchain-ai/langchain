{{ objname }}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

.. NOTE:: {{objname}} implements the standard :py:class:`Runnable Interface <langchain_core.runnables.base.Runnable>`. 🏃

    The :py:class:`Runnable Interface <langchain_core.runnables.base.Runnable>` has additional methods that are available on runnables, such as :py:meth:`with_config <langchain_core.runnables.base.Runnable.with_config>`, :py:meth:`with_types <langchain_core.runnables.base.Runnable.with_types>`, :py:meth:`with_retry <langchain_core.runnables.base.Runnable.with_retry>`, :py:meth:`assign <langchain_core.runnables.base.Runnable.assign>`, :py:meth:`bind <langchain_core.runnables.base.Runnable.bind>`, :py:meth:`get_graph <langchain_core.runnables.base.Runnable.get_graph>`, and more.

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
