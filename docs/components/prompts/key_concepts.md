# Key Concepts

## Prompts

A prompt is the input to a language model. It is a string of text that is used to generate a response from the language model.


## Prompt Templates

`PromptTemplates` are a way to create prompts in a reproducible way. They contain a template string, and a set of input variables. The template string can be formatted with the input variables to generate a prompt. The template string often contains instructions to the language model, a few shot examples, and a question to the language model.

`PromptTemplates` generically have a `format` method that takes in variables and returns a formatted string.
The most simple implementation of this is to have a template string with some variables in it, and then format it with the incoming variables.
More complex iterations dynamically construct the template string from few shot examples, etc.


## Few Shot Examples

Few shot examples refer to in-context examples that are provided to a language model as part of a prompt. The examples can be used to help the language model understand the context of the prompt, and as a result generate a better response. Few shot examples can contain both positive and negative examples about the expected response.

Below, we list out some few shot examples that may be relevant for the task of predicting the capital of a country.

```
Country: United States
Capital: Washington, D.C.

Country: Canada
Capital: Ottawa
```

## Example selection

Few shot examples are typically selected by the user. However, there are some cases where the few shot examples are selected automatically. For example, if the user is creating a prompt to predict the capital of a country, the few shot examples can be automatically selected from a list of countries and their capitals.
