## Summary

Fixes #34490

This PR fixes the `TypeError: Type is not msgpack serializable: Send` error that occurs when using `ShellToolMiddleware` with `InMemorySaver` checkpointer.

---

## Problem

When combining `ShellToolMiddleware` with `InMemorySaver()` checkpointer, the agent fails with a serialization error. Users reported that the middleware works fine without checkpointing, but fails when persistence is enabled.

**Error Message:**
```
TypeError: Type is not msgpack serializable: Send
```

**Reproduction:**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ShellToolMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    middleware=[ShellToolMiddleware(workspace="/tmp")],
    checkpointer=InMemorySaver(),
)
agent.invoke({"messages": [("user", "run ls")]})  # Fails!
```

---

## Root Cause Analysis

The issue is NOT that `Send` itself isn't serializable - LangGraph has proper msgpack support for `Send` via `EXT_CONSTRUCTOR_POS_ARGS`. The real problem is what's INSIDE `Send.arg`.

**The Flow:**
1. `ShellToolMiddleware` stores `_SessionResources` in agent state
2. `_SessionResources` contains non-serializable objects:
   - `subprocess.Popen` (shell process handle)
   - `threading.Thread` (output reader threads)
   - `queue.Queue` (output queues)
   - File handles (stdin/stdout/stderr)
   - `weakref.finalize` (cleanup handler)
3. In `model_to_tools()`, the edge function creates `Send("tools", ToolCallWithContext(state=state, ...))`
4. The full state dict (including `shell_session_resources`) is passed to `ToolCallWithContext`
5. When checkpointer tries to serialize the `Send` object, it fails on the non-serializable contents

**Why UntrackedValue annotation doesn't help:**
The `shell_session_resources` field is annotated with `UntrackedValue` and `PrivateStateAttr`, which should exclude it from checkpointing. However, when the entire state dict is passed into `Send.arg`, these annotations are bypassed - the whole dict gets serialized including the non-serializable fields.

---

## Solution

Filter out known non-serializable state fields before passing to `Send` in the `model_to_tools()` edge function.

**Changes in `libs/langchain_v1/langchain/agents/factory.py`:**

1. Added `_NON_SERIALIZABLE_STATE_FIELDS` constant - a frozenset of field names known to contain non-serializable middleware state

2. Added `_filter_serializable_state()` helper function - filters out non-serializable fields from state dict before passing to Send

3. Modified `model_to_tools()` edge - calls the filter before creating `ToolCallWithContext`

**Before:**
```python
Send(
    "tools",
    ToolCallWithContext(state=state, ...)  # state contains shell_session_resources
)
```

**After:**
```python
serializable_state = _filter_serializable_state(state)
Send(
    "tools",
    ToolCallWithContext(state=serializable_state, ...)  # filtered state
)
```

---

## Testing

All 28 existing shell tool tests pass:

```
tests/unit_tests/agents/middleware/implementations/test_shell_tool.py

test_executes_command_and_persists_state PASSED
test_restart_resets_session_environment PASSED
test_truncation_indicator_present PASSED
test_timeout_returns_error PASSED
test_redaction_policy_applies PASSED
test_startup_and_shutdown_commands PASSED
test_session_resources_finalizer_cleans_up PASSED
test_shell_tool_input_validation PASSED
test_normalize_shell_command_empty PASSED
test_normalize_env_non_string_keys PASSED
test_normalize_env_coercion PASSED
test_shell_tool_missing_command_string PASSED
test_tool_message_formatting_with_id PASSED
test_nonzero_exit_code_returns_error PASSED
test_truncation_by_bytes PASSED
test_startup_command_failure PASSED
test_shutdown_command_failure_logged PASSED
test_shutdown_command_timeout_logged PASSED
test_empty_output_replaced_with_no_output PASSED
test_stderr_output_labeling PASSED
test_normalize_commands_string_tuple_list[...] PASSED (4 variants)
test_async_methods_delegate_to_sync PASSED
test_shell_middleware_resumable_after_interrupt PASSED
test_get_or_create_resources_creates_when_missing PASSED
test_get_or_create_resources_reuses_existing PASSED

============================== 28 passed in 7.20s ==============================
```

---

## Backward Compatibility

**No breaking changes** - This fix only affects internal state serialization. Tool execution behavior is completely unchanged. The filtered fields are middleware-internal state that tools don't need access to.

---

## Design Decisions

**Why filter by field name instead of trying to serialize:**
1. **Performance** - Checking serializability would require attempting serialization on each field
2. **Predictability** - Known fields are deterministic; try/catch serialization could have edge cases
3. **Extensibility** - New non-serializable middleware state can be added to `_NON_SERIALIZABLE_STATE_FIELDS`

**Why filter at Send creation instead of in middleware:**
1. **Single point of fix** - All Send objects with state go through `model_to_tools()`
2. **Middleware independence** - Middlewares don't need to worry about serialization concerns
3. **Consistent behavior** - All tool dispatch follows the same pattern

---

## Future Considerations

1. **Framework-wide solution** - Consider adding automatic filtering based on `UntrackedValue`/`PrivateStateAttr` annotations at the serialization layer
2. **Other middlewares** - If new middlewares add non-serializable state, add their field names to `_NON_SERIALIZABLE_STATE_FIELDS`

---

## Related Issues

- [LangGraph #5891](https://github.com/langchain-ai/langgraph/issues/5891) - Same "Send" error with InjectedStore
- [LangGraph #5054](https://github.com/langchain-ai/langgraph/issues/5054) - ToolMessage serialization
- [LangGraph #5248](https://github.com/langchain-ai/langgraph/issues/5248) - AIMessage serialization
