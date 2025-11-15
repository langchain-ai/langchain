# Agent middleware dependency resolution

LangChain agents support **middleware chaining** that can intercept model and tool
execution. Middleware can now declare additional middleware dependencies and
ordering requirements, enabling reusable building blocks that assemble the
correct stack automatically.

This guide covers the dependency resolution process, the new
`MiddlewareSpec`/`OrderingConstraints` helpers, and tips for understanding the
resulting execution order.

## Declaring dependencies with `requires()`

Every `AgentMiddleware` subclass can override `requires()` and return a
sequence of `MiddlewareSpec` objects. Each spec describes another middleware
that should be inserted into the stack before the current middleware runs.

```python
from langchain.agents.middleware.types import AgentMiddleware, MiddlewareSpec


class RetryMiddleware(AgentMiddleware):
    def requires(self) -> list[MiddlewareSpec]:
        # Ensure requests are redacted before retries are attempted
        return [MiddlewareSpec(factory=PIIMiddleware)]
```

When `create_agent` resolves middleware, it recursively flattens all declared
dependencies. Each dependency runs before the middleware that requested it,
and the dependency's own `requires()` declarations are resolved as well.

## `MiddlewareSpec` reference

`MiddlewareSpec` encapsulates dependency metadata:

| Field | Description |
| --- | --- |
| `factory` / `middleware` | Nullary callable that creates the dependency, or a pre-instantiated middleware. One of the two must be provided. |
| `id` | Optional identifier override. Defaults to `middleware.id` or the class name. |
| `priority` | Optional numeric priority. Higher numbers run earlier when order ties cannot be broken by constraints or user list order. |
| `tags` | Optional sequence of tags for referencing in ordering constraints. |
| `ordering` | Optional `OrderingConstraints` specifying additional before/after requirements. |
| `merge_strategy` | Duplicate handling policy: `"first_wins"`, `"last_wins"`, or `"error"`. |

Dependencies that share an `id` are deduplicated according to their
`merge_strategy`:

- `first_wins` (default): keep the first instance and merge subsequent ordering
  constraints and tags.
- `last_wins`: replace the existing instance with the most recent dependency.
- `error`: raise a `ValueError` if another dependency with the same `id` is
  encountered. This matches the pre-existing behaviour for user supplied
  duplicates.

!!! note
    Middleware supplied without an explicit `id` (either by the user or within
    a dependency `MiddlewareSpec`) receives an auto-generated identifier. The
    first instance keeps its class name for backwards compatibility; additional
    instances of the same class gain a deterministic module-qualified suffix
    (for example, `my.module.Middleware#2`). This preserves historical behaviour
    where multiple differently configured instances of the same middleware class
    can coexist without triggering duplicate-id errors.

## `OrderingConstraints`

Ordering constraints ensure dependencies line up with other middleware:

```python
MiddlewareSpec(
    factory=AuthMiddleware,
    ordering=OrderingConstraints(
        after=("tag:session",),
        before=("retry-handler",),
    ),
)
```

- `after` accepts middleware identifiers or `tag:<tag-name>` references that
  must execute **before** the dependency.
- `before` accepts identifiers or tags that must execute **after** the
  dependency.
- Tag references apply the constraint to every middleware with that tag.

Self references are not allowed. Referencing an unknown id or tag raises a
`ValueError` during agent creation.

## Ordering semantics

Middleware resolution produces a deterministic order by applying the following
rules:

1. **User order is the starting point.** Middleware passed to `create_agent`
   retains its relative order whenever no other constraint applies.
2. **Dependencies run before their requestor.** Declared dependencies are
   inserted ahead of the middleware that required them.
3. **Before/after constraints add graph edges.** Constraints from
   `OrderingConstraints` augment the ordering graph by `id` or `tag`.
4. **Priority breaks ties.** When multiple nodes can execute next and the user
   order does not distinguish them, higher `priority` values win. If priorities
   are equal, the resolver uses the order dependencies were discovered.
5. **Cycles fail fast.** If the resulting graph contains a cycle, a
   `MiddlewareOrderCycleError` is raised with a human-readable cycle trace.

## Example

```python
from functools import partial

class AuditMiddleware(AgentMiddleware):
    id = "audit"
    tags = ("observability",)

    def requires(self) -> list[MiddlewareSpec]:
        return [
            MiddlewareSpec(
                factory=partial(RateLimitMiddleware, limit=10),
                merge_strategy="first_wins",
                ordering=OrderingConstraints(after=("tag:auth",)),
            )
        ]


class AuthMiddleware(AgentMiddleware):
    id = "auth"
    tags = ("auth",)


agent = create_agent(
    model="openai:gpt-4o",
    middleware=[AuthMiddleware(), AuditMiddleware()],
)
```

Resolution order:

1. `RateLimitMiddleware` (after everything tagged `auth`, before `audit`).
2. `AuthMiddleware` (user supplied).
3. `AuditMiddleware`.

If another middleware also requests the same `RateLimitMiddleware` with
`merge_strategy="first_wins"`, the resolver reuses the original instance and
adds the new ordering constraints.

## Troubleshooting

- **Duplicate id error** – Supply a unique `id` or configure `merge_strategy`
  (`"first_wins"`/`"last_wins"`).
- **Cycle detected** – Review the cycle path in the
  `MiddlewareOrderCycleError` message and adjust `before`/`after` constraints.
- **Unknown id/tag** – Ensure referenced identifiers are spelled correctly and
  that tagged middleware sets `tags` before resolution.
- **Unexpected ordering** – Remember that higher `priority` values preempt
  lower ones when no other constraints apply. Adjust `priority` or add explicit
  ordering constraints.
