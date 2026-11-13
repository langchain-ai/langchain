# Model-call budget

`ModelCallBudgetMiddleware(run_limit=8)` allows up to eight working model turns,
followed by one reserved, tool-free final response if the agent has not already
finished. Every model request receives a transient system-prompt addition reporting
used and remaining working turns. Consumption is also available as
`model_budget_used` in graph state. Existing message history is not modified.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallBudgetMiddleware

agent = create_agent(model, tools=tools, middleware=[ModelCallBudgetMiddleware(run_limit=8)])
```

The same middleware can be supplied to `create_deep_agent(middleware=[...])`.
Configure subagents separately; this is not an aggregate parent/subagent budget.

- Zero starts directly with the reserved final response. Negative or noninteger
  budgets are rejected.
- Each new invocation resets consumption; resuming an interrupted graph preserves
  the checkpointed counter.
- A working turn may execute multiple parallel tools. Their results are available
  to the final response. Tools that return directly can finish execution earlier.
- The final request removes tools, forced tool choice, and structured-output
  configuration. It produces plain text rather than a structured response. A
  provider that nevertheless returns tool calls causes an error before tool execution.
- Place this middleware last among model-request modifiers so another middleware
  does not reintroduce tools or structured output on the final turn.
- The unit is an agent model turn, not tokens, dollars, elapsed time, tool calls,
  provider retries, or auxiliary model calls made by other middleware. The final
  turn is additional to `run_limit`. This is not a billing cap or security boundary.
- A separate hard limiter or graph recursion limit can stop execution before the
  reserved turn; configure those limits to leave room for finalization.

Existing `ModelCallLimitMiddleware` remains unchanged: its `end` behavior returns
an artificial limit-exceeded message rather than asking the model to synthesize.
