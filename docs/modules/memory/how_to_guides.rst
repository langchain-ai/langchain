How-To Guides
=============

Types
-----

The first set of examples all highlight different types of memory.

`Buffer <./types/buffer.html>`_: How to use a type of memory that just keeps previous messages in a buffer.

`Buffer Window <./types/buffer_window.html>`_: How to use a type of memory that keeps previous messages in a buffer but only uses the previous `k` of them.

`Summary <./types/summary.html>`_: How to use a type of memory that summarizes previous messages.

`Summary Buffer <./types/summary_buffer.html>`_: How to use a type of memory that keeps a buffer of messages up to a point, and then summarizes them.

`Entity Memory <./types/entity_summary_memory.html>`_: How to use a type of memory that organizes information by entity.

`Knowledge Graph Memory <./types/kg.html>`_: How to use a type of memory that extracts and organizes information in a knowledge graph


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   ./types/*


Usage
-----

The examples here all highlight how to use memory in different ways.

`Adding Memory <./examples/adding_memory.html>`_: How to add a memory component to any single input chain.

`ChatGPT Clone <./examples/chatgpt_clone.html>`_: How to recreate ChatGPT with LangChain prompting + memory components.

`Adding Memory to Multi-Input Chain <./examples/adding_memory_chain_multiple_inputs.html>`_: How to add a memory component to any multiple input chain.

`Conversational Memory Customization <./examples/conversational_customization.html>`_: How to customize existing conversation memory components.

`Custom Memory <./examples/custom_memory.html>`_: How to write your own custom memory component.

`Adding Memory to Agents <./examples/agent_with_memory.html>`_: How to add a memory component to any agent.

`Conversation Agent <./examples/conversational_agent.html>`_: Example of a conversation agent, which combines memory with agents and a conversation focused prompt.

`Multiple Memory <./examples/multiple_memory.html>`_: How to use multiple types of memory in the same chain.


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   ./examples/*