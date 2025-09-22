# LangChain Learning Plan - Starting from the Basics

## Overview
This plan starts with the most basic LangChain building block: **PromptTemplate**. As a basic Python coder, we'll build understanding step by step with hands-on examples.

## Phase 1: Core Building Block - PromptTemplate (Start Here!)

### What is a PromptTemplate?
A PromptTemplate is the simplest and most fundamental component in LangChain. It's essentially a string template with variables that you can fill in with actual values.

**Think of it like this:**
- Regular string: `"Tell me a joke about cats"`
- PromptTemplate: `"Tell me a joke about {topic}"` where `{topic}` can be filled with any value

### Learning Steps

#### Step 1: Understanding the Basics (30 minutes)
**Files to read first:**
- `libs/core/langchain_core/prompts/prompt.py` (lines 1-100)
- `libs/core/tests/unit_tests/prompts/test_prompt.py` (lines 1-100)

**Key concepts to understand:**
1. What is a template vs a formatted string
2. Input variables - the `{variable_name}` parts
3. The `.format()` method - how to fill in the variables

#### Step 2: Hands-on Practice (45 minutes)
Create and run these examples:

```python
# Example 1: Basic PromptTemplate
from langchain_core.prompts import PromptTemplate

# Create a simple template
template = "Tell me a {adjective} joke about {topic}"
prompt = PromptTemplate.from_template(template)

# Use it
result = prompt.format(adjective="funny", topic="cats")
print(result)  # Output: "Tell me a funny joke about cats"
```

```python
# Example 2: More complex template
template = """
You are a {role}.
Please {task} about the topic: {topic}
Make it {style} and keep it under {length} words.
"""
prompt = PromptTemplate.from_template(template)

result = prompt.format(
    role="helpful assistant",
    task="write a summary",
    topic="Python programming",
    style="beginner-friendly",
    length="50"
)
print(result)
```

#### Step 3: Explore the Code (30 minutes)
**Run these debugging exercises:**

1. Print out the input variables:
```python
prompt = PromptTemplate.from_template("Hello {name}, welcome to {place}!")
print(prompt.input_variables)  # Should show: ['name', 'place']
```

2. Understand the template structure:
```python
print(prompt.template)  # See the raw template string
print(prompt.pretty_repr())  # See a pretty version
```

## Phase 2: Next Building Blocks (After mastering PromptTemplate)

### Step 4: Chat Templates (1-2 hours)
- **File:** `libs/core/langchain_core/prompts/chat.py`
- **Concept:** Templates for conversation-style prompts (system, human, AI messages)

### Step 5: LLM Basics (1-2 hours)
- **File:** `libs/core/langchain_core/language_models/base.py`
- **Concept:** How LangChain talks to language models

### Step 6: Combining Prompt + LLM (1 hour)
- **Concept:** Using prompts with actual language models

## Environment Setup

### Prerequisites
```bash
cd /Users/rohitsingh/Downloads/langchain
python -m venv .venv
source .venv/bin/activate
pip install -e "libs/core[test]"
```

### Create Practice Workspace
```bash
mkdir langchain_practice
cd langchain_practice
touch practice.py
```

## Learning Strategy for a Basic Python Coder

### 1. **Start Small**: Focus ONLY on PromptTemplate first
- Don't worry about LLMs, chains, agents yet
- Master the concept of templates and variable substitution

### 2. **Hands-on First**:
- Run every example
- Modify examples to see what breaks/changes
- Print everything to understand what's happening

### 3. **Read Tests**:
- Tests show you exactly how things are supposed to work
- Look at `test_prompt.py` for real usage examples

### 4. **One Concept at a Time**:
- Don't move to the next concept until you can:
  - Create a PromptTemplate from scratch
  - Use different template formats (f-string, mustache)
  - Debug when something goes wrong

## Success Criteria for Phase 1

You're ready to move on when you can:
- [ ] Create a PromptTemplate with multiple variables
- [ ] Use `.format()` to fill in the variables
- [ ] Understand what `input_variables` contains
- [ ] Create templates using both `from_template()` and the constructor
- [ ] Debug simple template errors (missing variables, etc.)

## Common Beginner Mistakes to Avoid

1. **Don't jump ahead**: Resist the urge to look at complex examples with agents/chains
2. **Always check input_variables**: Make sure you provide all required variables
3. **Print intermediate results**: Always print what your template looks like before and after formatting
4. **Start with simple examples**: Use single words as variables first, then move to complex text

## Next Steps After Phase 1

Once you master PromptTemplate, the next concepts build naturally:
- ChatPromptTemplate (for conversation)
- LLMs (to actually generate responses)
- Chains (to connect prompts and LLMs)
- Simple tools and utilities

**Remember**: The goal is deep understanding of fundamentals, not rushing through features!
