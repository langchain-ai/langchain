Agents
==========================

Some applications will require not just a predetermined chain of calls to LLMs/other tools,
but potentially an unknown chain that depends on the user input.
In these types of chains, there is a “agent” which has access to a suite of tools.
Depending on the user input, the agent can then decide which, if any, of these tools to call.

The following sections of documentation are provided:

- `Getting Started <agents/getting_started>`_: A notebook to help you get started working with agents as quickly as possible.

- `How-To Guides <agents/how_to_guides>`_: A collection of how-to guides. These highlight how to integrate various types of tools, how to work with different types of agent, and how to customize agents.

- `Reference </reference/modules/agents>`_: API reference documentation for all Agent classes.

- `Conceptual Guide <agents/conceptual_guide>`_: A conceptual guide going over the various concepts related to agents.


.. toctree::
   :maxdepth: 1
   :caption: Agents
   :name: Agents
   :hidden:

   agents/getting_started.ipynb
   agents/how_to_guides.rst
   agents/key_concepts.md
   /reference/modules/agents.rst