# Operations

The Operations module contains operations to manipulate and process thoughts represented by the [Thought](thought.py) class.  
Operations interface with a language model and use other helper classes like [Prompter](../prompter/prompter.py) and [Parser](../parser/parser.py) for effective communication and extraction of results from the language model.  
The [Graph of Operations](graph_of_operations.py) class is the main class of the module and is responsible for orchestrating the operations, defining their relationships and maintaining the state of the thought graph, also known as Graph Reasoning State.

## Graph of Operations
The [GraphOfOperations](graph_of_operations.py) class facilitates the creation and management of a directed graph representing the sequence and interrelationships of operations on thoughts. Here’s how you can construct and work with the Graph of Operations:

### Initialization
Creating a new instance of GraphOfOperations:

```python
from graph_of_thoughts.operations import GraphOfOperations

graph = GraphOfOperations()
```

Upon initialization, the graph will be empty with no operations, roots, or leaves.

### Adding Operations
**Append Operation:** You can append operations to the end of the graph using the append_operation method. This ensures that the operation becomes a successor to all current leaf operations in the graph.
```python
from graph_of_thoughts.operations import Generate

operationA = Generate()
graph.append_operation(operationA)
```
**Add Operation with Relationships:** If you want to define specific relationships for an operation, use the add_operation method.
```python
operationB = Generate()
operationB.predecessors.append(operationA)
graph.add_operation(operationB)
```
Remember to set up the predecessors (and optionally successors) for your operation before adding it to the graph.

## Available Operations
The following operations are available in the module:

**Score:** Collect all thoughts from preceding operations and score them either using the LLM or a custom scoring function.
- num_samples (Optional): The number of samples to use for scoring, defaults to 1.
- combined_scoring (Optional): Whether to score all thoughts together in a single prompt or separately, defaults to False.
- scoring_function (Optional): A function that takes in a list of thought states and returns a list of scores for each thought.

**ValidateAndImprove:** For each thought, validate it and if it is invalid, improve it.  
- num_samples (Optional): The number of samples to use for validation, defaults to 1.
- improve (Optional): Whether to improve the thought if it is invalid, defaults to True.
- num_tries (Optional): The number of times to try improving the thought, before giving up, defaults to 3.
- validate_function (Optional): A function that takes in a thought state and returns a boolean indicating whether the thought is valid.

**Generate:** Generate new thoughts from the current thoughts. If no previous thoughts are available, the thoughts are initialized with the input to the [Controller](../controller/controller.py).  
- num_branches_prompt (Optional): Number of responses that each prompt should generate (passed to prompter). Defaults to 1.
- num_branches_response (Optional): Number of responses the LLM should generate for each prompt. Defaults to 1.

**Improve:** Improve the current thoughts. This operation is similar to the ValidateAndImprove operation, but it does not validate the thoughts and always tries to improve them.  

**Aggregate:** Aggregate the current thoughts into a single thought. This operation is useful when you want to combine multiple thoughts into a single thought.  
- num_responses (Optional): Number of responses to request from the LLM (generates multiple new thoughts). Defaults to 1.

**KeepBestN:** Keep the best N thoughts from the preceding thoughts. Assumes that the thoughts are already scored and throws an error if they are not.
- n: The number of thoughts to keep in order of score.
- higher_is_better (Optional): Whether higher scores are better (True) or lower scores are better (False). Defaults to True.

**KeepValid:** Keep only the valid thoughts from the preceding thoughts. Assumes that each thought has already been validated, if not, it will be considered valid.

**Selector:** Select a number of thoughts from the preceding thoughts using a selection function. This is useful if subsequent operations should only be applied to a subset of the preceding thoughts.
- selector: A function that takes in a list of thoughts and returns a list of thoughts to select.

**GroundTruth**: Evaluates if the preceding/current thoughts solve the problem and equal the ground truth. This operation is useful for terminating the graph and checking if the final thoughts solve the problem, but is only useful if the ground truth is known.
- ground_truth_evaluator: A function that takes in a thought state and returns a boolean indicating whether the thought solves the problem.
