# LangChain PR Candidates

## Identified Issues / PR Opportunities

### 1. RePhraseQueryRetriever async stub (Bug - Good First Issue)
- **File**: `libs/langchain/langchain_classic/retrievers/re_phraser.py`
- **Issue**: #37619 - `_aget_relevant_documents` raises `NotImplementedError`
- **Impact**: Users calling `ainvoke()` on RePhraseQueryRetriever get crashes
- **Fix**: Implement async method mirroring sync logic using `ainvoke`
- **Complexity**: Small (~10 LOC)
- **Test**: No existing tests for re_phraser found

### 2. ChatAnthropic.bind_tools mutates caller dict (Bug)
- **File**: `libs/partners/anthropic/langchain_anthropic/chat_models.py` (line ~1859-1869)
- **Issue**: #37596 - `tool_choice` dict mutated in place when `parallel_tool_calls=False`
- **Impact**: Reusing same `tool_choice` dict causes unexpected `disable_parallel_tool_use` leakage
- **Fix**: Copy dict before mutation: `{**kwargs["tool_choice"], "disable_parallel_tool_use": ...}`
- **Complexity**: Small (~5 LOC)
- **Test**: Need to add test to `test_chat_models.py`

### 3. DeepSeek reasoning_content not preserved (Bug)
- **File**: `libs/partners/deepseek/langchain_deepseek/chat_models.py`
- **Issue**: #37178 - Multi-turn agent calls fail with 400 error
- **Impact**: reasoning_content from API response not sent back in subsequent requests
- **Complexity**: Medium (requires patching `_get_request_payload()`)

### 4. Missing logprobs support with Responses API (PR #36371)
- **File**: Partner OpenAI
- **Issue**: #36211 - Combination of `use_responses_api=True` and `logprobs=True` breaks
- **Status**: Open PR #36371 (size: S, new-contributor)
- **Complexity**: Small fix

### 5. Human-in-the-loop edit decision bug (Issue #33787)
- **File**: `libs/langchain/` (agent/hitl middleware)
- **Issue**: After edit decision, agent re-evaluates original message and makes duplicate tool call
- **Impact**: User's edited action is followed by original action, requiring rejection
- **Complexity**: Medium-High (requires state management change)

---

## Low-Hanging Fruit Candidates (Docs/Small Fixes)
1. README typos or outdated examples
2. Example code in docstrings that doesn't match current API
3. Missing type hints in some internal helper functions
4. Error message improvements for better debugging