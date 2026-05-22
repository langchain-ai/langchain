# LangChain - Selected 5 PR Plan

## Top 5 PR Candidates (Prioritized)

### Priority 1: Fix RePhraseQueryRetriever async NotImplementedError ✅ GOOD FIRST ISSUE
**Issue**: #37619  
**Type**: Bug fix  
**Package**: langchain-classic  
**File**: `libs/langchain/langchain_classic/retrievers/re_phraser.py`

**Problem**: 
`_aget_relevant_documents` at line 86-92 just raises `NotImplementedError`, while the sync method `_get_relevant_documents` at lines 61-84 has full implementation.

**Fix**:
```python
async def _aget_relevant_documents(
    self,
    query: str,
    *,
    run_manager: AsyncCallbackManagerForRetrieverRun,
) -> list[Document]:
    """Get relevant documents given a user question.

    Args:
        query: user question
        run_manager: callback handler to use

    Returns:
        Relevant documents for re-phrased question
    """
    re_phrased_question = await self.llm_chain.ainvoke(
        query,
        {"callbacks": run_manager.get_child()},
    )
    logger.info("Re-phrased question: %s", re_phrased_question)
    return await self.retriever.ainvoke(
        re_phrased_question,
        config={"callbacks": run_manager.get_child()},
    )
```

**Test Strategy**: Create `tests/unit_tests/retrievers/test_re_phrase_query_retriever.py` with:
- Mock LLM chain and inner retriever
- Test sync path via `_get_relevant_documents`
- Test async path via `_aget_relevant_documents`
- Verify callbacks are propagated correctly

**Complexity**: ~15 lines of code + tests

**PR Title**: `fix(langchain-classic): implement async `_aget_relevant_documents` for RePhraseQueryRetriever`

---

### Priority 2: Fix ChatAnthropic.bind_tools dict mutation
**Issue**: #37596  
**Type**: Bug fix  
**Package**: langchain-anthropic  
**File**: `libs/partners/anthropic/langchain_anthropic/chat_models.py` (~lines 1859-1864)

**Problem**: 
When `parallel_tool_calls=False` and `tool_choice` is a dict, the code mutates the caller's dict:
```python
kwargs["tool_choice"]["disable_parallel_tool_use"] = disable_parallel_tool_use
```

**Fix**:
Replace the existing tool_choice in kwargs with a copy before mutating:
```python
if "tool_choice" in kwargs:
    kwargs["tool_choice"] = {
        **kwargs["tool_choice"],
        "disable_parallel_tool_use": disable_parallel_tool_use,
    }
```

**Test Strategy**: Add test in `tests/unit_tests/test_chat_models.py`:
- Create dict with `tool_choice = {"type": "tool", "name": "GetWeather"}`
- Call `bind_tools` with `parallel_tool_calls=False`
- Assert original dict is unchanged

**Complexity**: ~5 lines

**PR Title**: `fix(anthropic): copy tool_choice dict before mutating in bind_tools`

---

### Priority 3: DeepSeek reasoning_content preservation
**Issue**: #37178  
**Type**: Bug fix  
**Package**: langchain-deepseek  
**File**: `libs/partners/deepseek/langchain_deepseek/chat_models.py`

**Problem**: 
Multi-turn agent calls fail with 400 because `reasoning_content` from API response is not sent back in subsequent requests.

**Fix**: 
Patch `_get_request_payload()` to extract and re-inject `reasoning_content` from original `AIMessage.additional_kwargs`.

**Complexity**: Medium (~20-30 lines)

**PR Title**: `fix(deepseek): preserve reasoning_content in multi-turn conversations`

---

### Priority 4: OpenAI logprobs with Responses API
**Issue**: #36211 (linked to PR #36371)  
**Type**: Bug fix  
**Package**: langchain-openai  
**Status**: Already has open PR #36371 (new-contributor, size: S)

**Notes**: 
- PR exists but not yet merged
- Could offer review help or similar fix in other partner packages
- See if same pattern affects other models

---

### Priority 5: Human-in-the-loop edit decision duplicate tool call
**Issue**: #33787  
**Type**: Bug fix  
**Package**: langchain  
**Area**: agents/human-in-the-loop middleware

**Problem**: 
After edit decision, agent re-evaluates original message and makes duplicate tool call requiring rejection.

**Complexity**: High (requires understanding agent state management)

**Notes**: 
- Good for deeper dive if time permits
- May touch multiple files in agent workflow

---

## Implementation Notes

### Conventional Commit Format
All PRs must follow: `type(scope): description`
- `fix(langchain-classic): ...`
- `fix(anthropic): ...`
- `fix(deepseek): ...`

### Test Requirements
- Unit tests must be added for any bug fix
- Tests go in `tests/unit_tests/` matching source structure
- No network calls in unit tests (use mocks)

### CI Requirements
Before submitting:
```bash
cd libs/langchain  # or relevant package
uv sync --group test
make format
make lint
make test
```