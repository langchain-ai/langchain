# Prompts

Prompts and all the tooling around them are integral to working with language models, and therefor
really important to get right, from both and interface and naming perspective. This is a "design doc"
of sorts explaining how we think about prompts and the related concepts, and why the interfaces
for working with are the way they are in LangChain

## Prompt

### Concept
A prompt is the final string that gets fed into the language model.

### LangChain Implementation
In LangChain a prompt is represented as just a string.

## Input Variables

### Concept
Input variables are parts of a prompt that are not known until runtime, eg could be user provided.

### LangChain Implementation
In LangChain input variables are just represented as a dictionary of key-value pairs, with the key
being the variable name and the value being the variable value.

## Examples

### Concept
Examples are basically datapoints that can be used to teach the model what to do. These can be included
in prompts to better instruct the model on what to do.

### LangChain Implementation
In LangChain examples are represented as a dictionary of key-value pairs, with the key being the feature
(or label) name, and the value being the feature (or label) value.

## Example Selector

### Concept
If you have a large number of examples, you may need to select which ones to include in the prompt. The
Example Selector is the class responsible for doing so.

### LangChain Implementation

#### BaseExampleSelector
In LangChain there is a BaseExampleSelector that exposes the following interface

```python
class BaseExampleSelector:
    
    def select_examples(self, examples: List[dict], input_variables: dict):
```

#### LengthExampleSelector
The LengthExampleSelector selects examples based on the length of the input variables. 
This is useful when you are worried about constructing a prompt that will go over the length
of the context window. For longer inputs, it will select fewer examples to include, while for
shorter inputs it will select more.

#### SemanticSimilarityExampleSelector
The SemanticSimilarityExampleSelector selects examples based on which examples are most similar
to the inputs. It does this by finding the examples with the embeddings that have the greatest 
cosine similarity with the inputs.


## Prompt Template

### Concept
The prompts that get fed into the language model are nearly always not hardcoded, but rather a combination
of parts, including Examples and Input Variables. A prompt template is responsible
for taking those parts and constructing a prompt.

### LangChain Implementation

#### BasePromptTemplate
In LangChain there is a BasePromptTemplate that exposes the following interface

```python
class BasePromptTemplate:
    
    @property
    def input_variables(self) -> List[str]:
        
    def format(self, **kwargs) -> str:
```
The input variables property is used to provide introspection of the PromptTemplate and know
what inputs it expects. The format method takes in input variables and returns the prompt.

#### PromptTemplate
The PromptTemplate implementation is the most simple form of a prompt template. It consists of three parts:
- input variables: which variables this prompt template expects
- template: the template into which these variables will be formatted
- template format: the format of the template (eg mustache, python f-strings, etc)

For example, if I was making an application that took a user inputted concept and asked a language model
to make a joke about that concept, I might use this specification for the PromptTemplate
- input variables = "thing"
- template = "Tell me a joke about {thing}"
- template format = "f-string"

#### FewShotPromptTemplate
A FewShotPromptTemplate is a Prompt Template that includes some examples. It consists of:
- examples: a list of examples to use
- example prompt template: a Prompt Template responsible for taking an individual example (a dictionary) and turning it into a string to be used in the prompt.
- example selector: an Example Selector to select which examples to use
- prefix: the template put in the prompt before listing any examples
- suffix: the template put in the prompt after listing any examples
- example separator: a string separator which is used to join the prefix, the examples, and the suffix together
