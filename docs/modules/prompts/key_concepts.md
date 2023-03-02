# Key Concepts

## Prompts

A prompt is the input to a language model. It is a string of text that is used to generate a response from the language model.

## Prompt Templates

`PromptTemplates` are a way to create prompts in a reproducible way. They contain a template string, and a set of input variables. The template string can be formatted with the input variables to generate a prompt. The template string often contains instructions to the language model, a few shot examples, and a question to the language model.

`PromptTemplates` generically have a `format` method that takes in variables and returns a formatted string.
The most simple implementation of this is to have a template string with some variables in it, and then format it with the incoming variables.
More complex iterations dynamically construct the template string from few shot examples, etc.

To learn more about `PromptTemplates`, see [Prompt Templates](getting_started.md).

As an example, consider the following template string:

```python
"""
Predict the capital of a country.

Country: {country}
Capital:
"""
```

### Input Variables

Input variables are the variables that are used to fill in the template string. In the example above, the input variable is `country`.

Given an input variable, the `PromptTemplate` can generate a prompt by filling in the template string with the input variable. For example, if the input variable is `United States`, the template string can be formatted to generate the following prompt:

```python
"""
Predict the capital of a country.

Country: United States
Capital:
"""
```

## Few Shot Examples

Few shot examples refer to in-context examples that are provided to a language model as part of a prompt. The examples can be used to help the language model understand the context of the prompt, and as a result generate a better response. Few shot examples can contain both positive and negative examples about the expected response.

Below, we list out some few shot examples that may be relevant for the task of predicting the capital of a country.

```
Country: United States
Capital: Washington, D.C.

Country: Canada
Capital: Ottawa
```

To learn more about how to provide few shot examples, see [Few Shot Examples](examples/few_shot_examples.ipynb).

<!-- TODO(shreya): Add correct link here. -->

## Example selection

If there are multiple examples that are relevant to a prompt, it is important to select the most relevant examples. Generally, the quality of the response from the LLM can be significantly improved by selecting the most relevant examples. This is because the language model will be able to better understand the context of the prompt, and also potentially learn failure modes to avoid.

To help the user with selecting the most relevant examples, we provide example selectors that select the most relevant based on different criteria, such as length, semantic similarity, etc. The example selector takes in a list of examples and returns a list of selected examples, formatted as a string. The user can also provide their own example selector. To learn more about example selectors, see [Example Selection](example_selection.md).

<!-- TODO(shreya): Add correct link here. -->

## Serialization

To make it easy to share `PromptTemplates`, we provide a `serialize` method that returns a JSON string. The JSON string can be saved to a file, and then loaded back into a `PromptTemplate` using the `deserialize` method. This allows users to share `PromptTemplates` with others, and also to save them for later use.

To learn more about serialization, see [Serialization](examples/prompt_serialization.ipynb).

<!-- TODO(shreya): Provide correct link. -->
