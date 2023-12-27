"""**OutputParser** classes parse the output of an LLM call.

**Class hierarchy:**

.. code-block::

    BaseLLMOutputParser --> BaseOutputParser --> <name>Parser  # OutputFunctionsParser, PydanticOutputParser
                                             --> BaseTransformOutputParser --> 
                                                           BaseCumulativeTransformOutputParser --> <name>Parser
                        --> BaseGenerationOutputParser --> <name>Parser

**Main helpers:**

.. code-block::

    Generation, ChatGeneration
"""  # noqa: E501
