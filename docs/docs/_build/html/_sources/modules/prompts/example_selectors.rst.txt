Example Selectors
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/prompts/example-selectors>`_


If you have a large number of examples, you may need to select which ones to include in the prompt. The ExampleSelector is the class responsible for doing so.

The base interface is defined as below::

    class BaseExampleSelector(ABC):
        """Interface for selecting examples to include in prompts."""

        @abstractmethod
        def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
            """Select which examples to use based on the inputs."""


The only method it needs to expose is a ``select_examples`` method. This takes in the input variables and then returns a list of examples. It is up to each specific implementation as to how those examples are selected. Let's take a look at some below.

See below for a list of example selectors.


.. toctree::
   :maxdepth: 1
   :glob:

   ./example_selectors/examples/*