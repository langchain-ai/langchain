# LangChain Learning Practice

## Quick Start

1. **Set up environment:**
```bash
cd /Users/rohitsingh/Downloads/langchain
python -m venv .venv
source .venv/bin/activate
pip install -e "libs/core[test]"
```

2. **Test your setup:**
```bash
cd langchain_practice
python setup_check.py
```

3. **Start learning:**
```bash
python step1_basic_prompts.py
```

## Learning Path

1. **`setup_check.py`** - Verify your environment works
2. **`step1_basic_prompts.py`** - Learn PromptTemplate basics
3. **`step2_advanced_prompts.py`** - Advanced prompt features

## Files Created

- **`LEARNING_PLAN.md`** - Detailed learning strategy
- **Practice files** - Hands-on exercises with explanations
- **Setup check** - Environment validation

## Key Concept: Start with PromptTemplate

PromptTemplate is the most basic building block in LangChain. It's like a Mad Libs template:

```python
from langchain_core.prompts import PromptTemplate

# Create template
prompt = PromptTemplate.from_template("Tell me a {adjective} joke about {topic}")

# Use it
result = prompt.format(adjective="funny", topic="cats")
# Result: "Tell me a funny joke about cats"
```

**Master this first** before moving to other LangChain concepts!
