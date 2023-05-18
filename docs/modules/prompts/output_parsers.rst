Output Parsers
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/prompts/output-parser>`_


Language models output text. But many times you may want to get more structured information than just text back. This is where output parsers come in.

Output parsers are classes that help structure language model responses. There are two main methods an output parser must implement:

- ``get_format_instructions() -> str``: A method which returns a string containing instructions for how the output of a language model should be formatted.
- ``parse(str) -> Any``: A method which takes in a string (assumed to be the response from a language model) and parses it into some structure.

And then one optional one:

- ``parse_with_prompt(str) -> Any``: A method which takes in a string (assumed to be the response from a language model) and a prompt (assumed to the prompt that generated such a response) and parses it into some structure. The prompt is largely provided in the event the OutputParser wants to retry or fix the output in some way, and needs information from the prompt to do so.

To start, we recommend familiarizing yourself with the Getting Started section

.. toctree::
   :maxdepth: 1

   ./output_parsers/getting_started.md

After that, we provide deep dives on all the different types of output parsers.

.. toctree::
   :maxdepth: 1
   :glob:

   ./output_parsers/examples/*