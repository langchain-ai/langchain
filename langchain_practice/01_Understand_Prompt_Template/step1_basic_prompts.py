#!/usr/bin/env python3
"""
Step 1: Basic PromptTemplate Practice
=====================================

This file contains simple exercises to understand PromptTemplate basics.
Run each section and observe the output to build understanding.
"""

from langchain_core.prompts import PromptTemplate

print("=== LANGCHAIN LEARNING: Step 1 - Basic Prompts ===\n")

# Exercise 1: Your first PromptTemplate
print("Exercise 1: Basic PromptTemplate")
print("-" * 40)

# Method 1: Using from_template (easiest way)
template_string = "Hello {name}, welcome to {place}!"
prompt = PromptTemplate.from_template(template_string)

print(f"Template: {prompt.template}")
print(f"Input variables: {prompt.input_variables}")

# Format the template with actual values
result = prompt.format(name="Rohit", place="LangChain")
print(f"Formatted result: {result}")
print()

# Exercise 2: Multiple variables and complex templates
print("Exercise 2: More Complex Template")
print("-" * 40)

template = """
You are a {role} helping someone learn {subject}.
Please explain {topic} in a {style} way.
Keep your explanation under {word_limit} words.
"""

prompt2 = PromptTemplate.from_template(template)
print(f"Input variables: {prompt2.input_variables}")

result2 = prompt2.format(
    role="friendly teacher",
    subject="programming",
    topic="variables",
    style="simple",
    word_limit="50"
)
print("Formatted result:")
print(result2)
print()

# Exercise 3: Understanding what happens behind the scenes
print("Exercise 3: Behind the Scenes")
print("-" * 40)

# This is what LangChain does automatically when you use from_template
manual_prompt = PromptTemplate(
    input_variables=["emotion", "animal"],
    template="Tell me a {emotion} story about a {animal}"
)

print(f"Manually created prompt template: {manual_prompt.template}")
print(f"Manual input variables: {manual_prompt.input_variables}")

result3 = manual_prompt.format(emotion="happy", animal="dog")
print(f"Manual formatting result: {result3}")
print()

# Exercise 4: What happens when you miss a variable?
print("Exercise 4: Error Handling")
print("-" * 40)

try:
    # This will cause an error - we're missing the 'animal' variable
    bad_result = manual_prompt.format(emotion="sad")
    print(f"This shouldn't print: {bad_result}")
except KeyError as e:
    print(f"Error caught! Missing variable: {e}")
    print("Lesson: You must provide ALL variables in input_variables")
print()

# Exercise 5: Pretty printing for debugging
print("Exercise 5: Debugging Tools")
print("-" * 40)

debug_template = "Write a {length} {type} about {topic} for {audience}"
debug_prompt = PromptTemplate.from_template(debug_template)

print("Template info:")
print(f"  Raw template: {debug_prompt.template}")
print(f"  Variables needed: {debug_prompt.input_variables}")
print(f"  Pretty version: {debug_prompt.pretty_repr()}")
print()

# YOUR TURN: Try modifying the examples above!
print("=== YOUR PRACTICE AREA ===")
print("Try creating your own PromptTemplate below:")
print()

# TODO: Create your own PromptTemplate here
# Uncomment and modify these lines:
# my_template = "..."
# my_prompt = PromptTemplate.from_template(my_template)
# my_result = my_prompt.format(...)
# print(my_result)

print("\n=== NEXT STEPS ===")
print("1. Run this file: python step1_basic_prompts.py")
print("2. Modify the templates and variables to see what happens")
print("3. Create your own template in the practice area")
print("4. When comfortable, move to step2_advanced_prompts.py")
