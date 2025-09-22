#!/usr/bin/env python3
"""
Environment Setup Check
=======================

Run this first to make sure your environment is set up correctly.
"""

print("=== LANGCHAIN ENVIRONMENT CHECK ===\n")

# Check 1: Can we import LangChain core?
try:
    from langchain_core.prompts import PromptTemplate
    print("✅ LangChain core import successful")
except ImportError as e:
    print(f"❌ LangChain core import failed: {e}")
    print("Run: pip install -e 'libs/core[test]' from the langchain directory")
    exit(1)

# Check 2: Can we create a basic prompt?
try:
    prompt = PromptTemplate.from_template("Hello {name}!")
    result = prompt.format(name="World")
    print(f"✅ Basic PromptTemplate works: '{result}'")
except Exception as e:
    print(f"❌ PromptTemplate creation failed: {e}")
    exit(1)

# Check 3: Check Python version
import sys
python_version = sys.version_info
print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version < (3, 8):
    print("⚠️  Warning: Python 3.8+ recommended for LangChain")

# Check 4: Check working directory
import os
current_dir = os.getcwd()
print(f"✅ Current directory: {current_dir}")

if "langchain" not in current_dir.lower():
    print("⚠️  Warning: Make sure you're in the langchain directory")

print("\n=== SETUP COMPLETE ===")
print("You're ready to start learning!")
print("Run: python step1_basic_prompts.py")
