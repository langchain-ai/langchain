## Description
This PR fixes the OutputParserException that occurs when using Ollama models instead of Gemini models.

## Problem
When switching from Gemini to Ollama, users encountered OutputParserException due to different response formats between the two LLM providers. The output parser was expecting Gemini-specific formatting.

## Solution
- Updated the output parser to handle multiple response formats
- Added fallback parsing logic for Ollama responses
- Improved error messages for debugging
- Added validation for different LLM response structures

## Changes Made
- Modified `langchain/output_parsers/base.py` to handle Ollama response format
- Updated error handling in the parser chain
- Added unit tests for Ollama compatibility
- Updated documentation examples

## Testing
- [ ] Existing tests pass
- [ ] Added new tests for Ollama compatibility
- [ ] Manually tested with both Gemini and Ollama models
- [ ] Verified backward compatibility

## Fixes
Closes #33016

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective
- [ ] New and existing unit tests pass locally with my changes
