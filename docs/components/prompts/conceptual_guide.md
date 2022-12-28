# Conceptual Guide

## PromptTemplates
PromptTemplates generically have a `format` method that takes in variables and returns a formatted string.
The most simple implementation of this is to have a template string with some variables in it, and then format it with the incoming variables.
More complex iterations dynamically construct the template string from few shot examples, etc.
