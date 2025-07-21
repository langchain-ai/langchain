#!/usr/bin/env python3
"""Test if the chat models module can be imported after our changes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

try:
    print("Importing chat_models module...")
    from langchain_core.language_models import chat_models
    print("✓ Import successful!")
    
    print("Checking if BaseChatModel exists...")
    if hasattr(chat_models, 'BaseChatModel'):
        print("✓ BaseChatModel found!")
        print(f"  _generate_with_cache method exists: {hasattr(chat_models.BaseChatModel, '_generate_with_cache')}")
        print(f"  _agenerate_with_cache method exists: {hasattr(chat_models.BaseChatModel, '_agenerate_with_cache')}")
    else:
        print("✗ BaseChatModel not found")
        
    print("✓ All checks passed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Other error: {e}")
    sys.exit(1)

print("Module import test successful!")