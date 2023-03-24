How-To Guides
=============

If you're new to the library, you may want to start with the `Quickstart <./getting_started.html>`_.

The user guide here shows more advanced workflows and how to use the library in different ways.

`Pydantic Output Parser <./examples/pydantic.html>`_: How to use a Pydantic object to define, parse, and validate output format.

`Output Fixing Parser <./examples/output_fixing_parser.html>`_: How to use the OutputFixingParser - an output parser which wraps another output parser and retries the formatting if necessary.

`Retry Output Parser <./examples/retry.html>`_: How to use the RetryOutput - an output parser which wraps another output parser and retries the formatting if necessary.

`Structured Output Parser <./examples/structured.html>`_: How to use a simpler output parser, that is most relevant when you just want multiple string outputs.

`Comma Separated Output Parser <./examples/comma_separated.html>`_: How to use a simpler output parsers, which is most relevant when you just want a list to be returned.



.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   ./examples/*
