Agents
==========================

Some applications will require not just a predetermined chain of calls to LLMs/other tools,
but potentially an unknown chain that depends on the user input.
In these types of chains, there is a “agent” which has access to a suite of tools.
Depending on the user input, the agent can then decide which, if any, of these tools to call.

The following sections of documentation are provided:

- `Getting Started <./agents/getting_started.html>`_: A notebook to help you get started working with agents as quickly as possible.

- `Key Concepts <./agents/key_concepts.html>`_: A conceptual guide going over the various concepts related to agents.

- `How-To Guides <./agents/how_to_guides.html>`_: A collection of how-to guides. These highlight how to integrate various types of tools, how to work with different types of agent, and how to customize agents.

- `Reference <../reference/modules/agents.html>`_: API reference documentation for all Agent classes.



.. toctree::
   :maxdepth: 1
   :caption: Agents
   :name: Agents
   :hidden:

   ./agents/getting_started.ipynb
   ./agents/key_concepts.md
   ./agents/how_to_guides.rst
   Reference<../reference/modules/agents.rst>