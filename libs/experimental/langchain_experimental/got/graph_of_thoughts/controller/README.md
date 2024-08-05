# Controller

The Controller class is responsible for traversing the Graph of Operations (GoO), which is a static structure that is constructed once, before the execution starts.
GoO prescribes the execution plan of thought operations and the Controller invokes their execution, generating the Graph Reasoning State (GRS). 

In order for a GoO to be executed, an instance of Large Language Model (LLM) must be supplied to the controller (along with other required objects).
Please refer to the [Language Models](../language_models/README.md) section for more information about LLMs. 

The following section describes how to instantiate the Controller to run a defined GoO. 

## Controller Instantiation
- Requires custom `Prompter`, `Parser`, as well as instantiated `GraphOfOperations` and `AbstractLanguageModel` - creation of these is described separately.
- Prepare initial state (thought) as dictionary - this can be used in the initial prompts by the operations.
```
lm = ...create
graph_of_operations = ...create

executor = controller.Controller(
    lm,
    graph_of_operations,
    <CustomPrompter()>,
    <CustomParser()>,
    <initial state>,
)
executor.run()
executor.output_graph("path/to/output.json")
```
- After the run the graph is written to an output file, which contains individual operations, their thoughts, information about scores and validity and total amount of used tokens / cost.
