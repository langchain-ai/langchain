#!/usr/bin/env python3
"""
Step 2: Advanced PromptTemplate Features
=========================================

This builds on step1 with more advanced PromptTemplate features.
Only run this after you're comfortable with step1_basic_prompts.py
"""

from langchain_core.prompts import PromptTemplate

print("=== LANGCHAIN LEARNING: Step 2 - Advanced Prompts ===\n")

# Exercise 1: Different template formats
print("Exercise 1: Template Formats")
print("-" * 40)

# Default format (f-string style)
f_string_template = "Hello {name}, today is {day}"
f_prompt = PromptTemplate.from_template(f_string_template)
print(f"F-string result: {f_prompt.format(name='Bob', day='Monday')}")

# Mustache format (different syntax)
mustache_template = "Hello {{name}}, today is {{day}}"
mustache_prompt = PromptTemplate.from_template(
    mustache_template,
    template_format="mustache"
)
print(f"Mustache result: {mustache_prompt.format(name='Bob', day='Monday')}")
print()

# Exercise 2: Partial templates (pre-filling some variables)
print("Exercise 2: Partial Templates")
print("-" * 40)

base_template = "Write a {length} {type} about {topic}"
base_prompt = PromptTemplate.from_template(base_template)

# Create a partial template with some variables pre-filled
essay_prompt = base_prompt.partial(type="essay", length="short")
print(f"Partial template variables: {essay_prompt.input_variables}")
print(f"Only need to provide: {essay_prompt.input_variables}")

# Now we only need to provide the remaining variable
result = essay_prompt.format(topic="artificial intelligence")
print(f"Partial result: {result}")
print()

# Exercise 3: Callable partials (dynamic values)
print("Exercise 3: Dynamic Partials")
print("-" * 40)

from datetime import datetime

# Function that gets called each time the template is used
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_template = "Report generated at {timestamp}: {content}"
time_prompt = PromptTemplate.from_template(time_template)

# Partial with a function - gets called each time format() is used
dynamic_prompt = time_prompt.partial(timestamp=get_current_time)

# Notice how timestamp is different each time
result1 = dynamic_prompt.format(content="First report")
print(f"First call: {result1}")

import time
time.sleep(1)  # Wait a second

result2 = dynamic_prompt.format(content="Second report")
print(f"Second call: {result2}")
print()

# Exercise 4: Template validation
print("Exercise 4: Template Validation")
print("-" * 40)

# LangChain automatically detects variables in your template
auto_detected = PromptTemplate.from_template(
    "Process {data} using {method} and save to {output_file}"
)
print(f"Auto-detected variables: {auto_detected.input_variables}")

# You can also manually specify (useful for validation)
try:
    manual_validation = PromptTemplate(
        template="Hello {name}",
        input_variables=["name", "age"]  # We said age but it's not in template
    )
except ValueError as e:
    print(f"Validation error: {e}")

print()

# Exercise 5: Complex real-world example
print("Exercise 5: Real-world Template")
print("-" * 40)

# A template you might actually use for code generation
code_template = """
Generate a {language} function that:
- Function name: {function_name}
- Purpose: {purpose}
- Input parameters: {parameters}
- Return type: {return_type}
- Style: {style}

Make sure to include:
1. Proper documentation
2. Error handling
3. Type hints (if applicable)
"""

code_prompt = PromptTemplate.from_template(code_template)

result = code_prompt.format(
    language="Python",
    function_name="calculate_average",
    purpose="Calculate the average of a list of numbers",
    parameters="numbers: List[float]",
    return_type="float",
    style="clean and readable"
)

print("Generated prompt:")
print(result)
print()

# Exercise 6: Debugging helpers
print("Exercise 6: Debugging Helpers")
print("-" * 40)

debug_template = "Analyze {data_type} from {source} using {algorithm}"
debug_prompt = PromptTemplate.from_template(debug_template)

# Useful debugging methods
print(f"Template string: {debug_prompt.template}")
print(f"Required variables: {debug_prompt.input_variables}")
print(f"Template type: {debug_prompt._prompt_type}")

# Pretty print shows you what the template looks like with placeholders
print("Pretty representation:")
print(debug_prompt.pretty_repr())
print()

print("=== YOUR PRACTICE AREA ===")
print("Try these challenges:")
print("1. Create a template with 5+ variables")
print("2. Make a partial template with some variables pre-filled")
print("3. Create a template for generating email templates")
print("4. Use a callable partial to include current date/time")
print()

# TODO: Your practice code here
# Challenge 1: Multi-variable template
# challenge_template = "..."

# Challenge 2: Partial template
# partial_prompt = your_prompt.partial(...)

# Challenge 3: Email template
# email_template = "..."

print("\n=== NEXT STEPS ===")
print("1. Master these advanced features")
print("2. Try the challenges above")
print("3. When ready, move to understanding LLMs (language models)")
print("4. Then learn how to connect prompts with LLMs")
