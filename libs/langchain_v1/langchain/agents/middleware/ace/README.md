# ACE Middleware

**ACE (Agentic Context Engineering)** is a middleware that enables LangChain agents to self-improve by maintaining an evolving "playbook" of strategies, insights, and patterns learned from interactions.

Based on the research paper: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)

## Overview

Traditional agents use static prompts that don't learn from experience. ACE solves this by:

1. **Injecting an evolving playbook** into the system prompt before each model call
2. **Analyzing agent trajectories** after each response to identify what worked and what didn't
3. **Curating the playbook** periodically to add new insights and prune harmful advice

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Loop                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│   │   Playbook   │───▶│    Model     │───▶│   Response   │      │
│   │  (injected)  │    │    Call      │    │              │      │
│   └──────────────┘    └──────────────┘    └──────┬───────┘      │
│          ▲                                        │              │
│          │                                        ▼              │
│   ┌──────┴───────┐    ┌──────────────┐    ┌──────────────┐      │
│   │   Curator    │◀───│  Reflector   │◀───│  Trajectory  │      │
│   │  (periodic)  │    │  (analyze)   │    │   Analysis   │      │
│   └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Design

### Three-Role Architecture

ACE uses three specialized roles that work together:

| Role | Purpose | When It Runs |
|------|---------|--------------|
| **Generator** | Uses playbook to enhance responses | Every model call (via `wrap_model_call`) |
| **Reflector** | Analyzes trajectories and tags bullets | After each model response (`after_model`) |
| **Curator** | Adds new insights to playbook | Every N interactions (`curator_frequency`) |

### The Playbook

The playbook is a structured document containing bullets organized by section:

```
## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: Always verify data types before processing
[str-00002] helpful=3 harmful=1 :: Consider edge cases in financial data

## FORMULAS & CALCULATIONS
[cal-00001] helpful=8 harmful=0 :: NPV = Σ(Cash Flow / (1+r)^t)

## COMMON MISTAKES TO AVOID
[mis-00001] helpful=6 harmful=0 :: Don't forget timezone conversions
```

Each bullet has:
- **ID**: `[section-00000]` format for tracking
- **Counts**: `helpful=X harmful=Y` updated by the reflector
- **Content**: The actual advice or strategy

### Middleware Hooks

ACE implements the following middleware hooks:

```python
class ACEMiddleware(AgentMiddleware[ACEState, Any]):

    def before_agent(self, state, runtime):
        """Initialize playbook state at start of agent execution."""

    def wrap_model_call(self, request, handler):
        """Inject playbook into system prompt before each model call."""

    def after_model(self, state, runtime):
        """Run reflector to analyze trajectory and update bullet counts."""
```

## How Trajectories Update the Playbook

The key innovation of ACE is using **agent trajectories** (the sequence of reasoning and actions) to continuously improve the playbook. Here's how it works:

### 1. Trajectory Capture

When the agent responds, the trajectory includes:
- The user's question/query
- The agent's reasoning and response
- Which playbook bullets were referenced (extracted via `[xxx-00000]` patterns)

### 2. Reflection Analysis

The **Reflector** model analyzes the trajectory in `after_model()`:

```python
# Reflector receives:
- question: "What is the NPV of this investment?"
- reasoning_trace: "Using [cal-00001], I calculated..."
- feedback: "Response was correct" or error information
- bullets_used: "[cal-00001] helpful=8 harmful=0 :: NPV formula"
```

The reflector outputs:
```json
{
  "analysis": "The agent correctly applied the NPV formula",
  "what_worked": "Referenced the correct formula bullet",
  "what_failed": "Nothing significant",
  "key_insight": "Always verify discount rate units match cash flow periods",
  "bullet_tags": [
    {"id": "cal-00001", "tag": "helpful"}
  ]
}
```

### 3. Bullet Count Updates

Based on reflector tags, bullet counts are updated:

```
Before: [cal-00001] helpful=8 harmful=0 :: NPV formula
After:  [cal-00001] helpful=9 harmful=0 :: NPV formula
```

This creates a **feedback signal** that tracks which bullets are actually useful.

### 4. Periodic Curation

Every `curator_frequency` interactions, the **Curator** model:
- Reviews recent reflections
- Identifies new insights worth adding
- Generates `ADD` operations for new bullets

```json
{
  "reasoning": "The reflection identified a new insight about discount rates",
  "operations": [
    {
      "type": "ADD",
      "section": "COMMON MISTAKES TO AVOID",
      "content": "Always verify discount rate units match cash flow periods",
      "reason": "Derived from recent reflection"
    }
  ]
}
```

### 5. Automatic Pruning (Optional)

When `auto_prune=True`, bullets with high harmful-to-helpful ratios are removed:

```python
# Bullet with harmful > helpful gets pruned
[str-00005] helpful=1 harmful=5 :: Bad advice  # REMOVED
```

## Running the Demo

```bash
cd libs/langchain_v1/langchain/agents/middleware/ace
uv run --env-file .env python examples/ace_playbook_demo.py
```

This runs an interactive demo that shows the playbook evolving as the agent solves math problems.

## Usage Example

### Basic Usage

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ACEMiddleware
from langchain_core.messages import HumanMessage

# Create ACE middleware with reflection enabled
ace = ACEMiddleware(
    reflector_model="gpt-4o-mini",  # Analyzes trajectories
    curator_frequency=10,            # Curate every 10 interactions
)

# Create agent with ACE middleware
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[ace],
)

# Use the agent - it will self-improve over time
result = agent.invoke({
    "messages": [HumanMessage(content="Calculate the NPV of...")]
})
```

### With Custom Initial Playbook

```python
ace = ACEMiddleware(
    reflector_model="gpt-4o-mini",
    curator_model="gpt-4o-mini",
    curator_frequency=5,
    initial_playbook="""## STRATEGIES & INSIGHTS
[str-00001] helpful=10 harmful=0 :: Always verify data types before processing
[str-00002] helpful=5 harmful=0 :: Break complex problems into smaller steps

## COMMON MISTAKES TO AVOID
[mis-00001] helpful=8 harmful=0 :: Don't forget to handle timezone conversions
[mis-00002] helpful=6 harmful=0 :: Validate user inputs before processing
""",
)
```

### With Auto-Pruning

```python
ace = ACEMiddleware(
    reflector_model="gpt-4o-mini",
    curator_model="gpt-4o-mini",
    curator_frequency=10,
    auto_prune=True,           # Enable automatic pruning
    prune_threshold=0.5,       # Prune if harmful/(helpful+harmful) > 0.5
    prune_min_interactions=5,  # Only prune after 5+ interactions
)
```

### Full Configuration

```python
ace = ACEMiddleware(
    # Models for reflection and curation
    reflector_model="gpt-4o-mini",      # Or pass a BaseChatModel instance
    curator_model="gpt-4o-mini",        # Defaults to reflector_model if not set

    # Initial playbook content
    initial_playbook=None,              # Uses default template if not provided

    # Curation settings
    curator_frequency=5,                # Run curator every N interactions
    playbook_token_budget=80000,        # Max tokens for playbook

    # Feature toggles
    enable_reflection=True,             # Run reflector after model calls
    enable_curation=True,               # Run curator periodically

    # Pruning settings
    auto_prune=False,                   # Auto-remove harmful bullets
    prune_threshold=0.5,                # Harmful ratio threshold
    prune_min_interactions=3,           # Min interactions before pruning
)
```

## State Schema

ACE extends the standard `AgentState` with additional fields:

```python
class ACEState(AgentState):
    # The evolving playbook (private, not exposed in input/output)
    ace_playbook: dict[str, Any]        # {content, next_global_id, stats}

    # Last reflection for context in next iteration
    ace_last_reflection: str

    # Counter for curator frequency
    ace_interaction_count: int
```

## Playbook Sections

The default playbook template includes these sections:

| Section | Slug | Purpose |
|---------|------|---------|
| STRATEGIES & INSIGHTS | `str` | General approaches and tactics |
| FORMULAS & CALCULATIONS | `cal` | Mathematical formulas and computation patterns |
| CODE SNIPPETS & TEMPLATES | `cod` | Reusable code patterns |
| COMMON MISTAKES TO AVOID | `mis` | Known pitfalls and anti-patterns |
| PROBLEM-SOLVING HEURISTICS | `heu` | Decision-making rules of thumb |
| CONTEXT CLUES & INDICATORS | `ctx` | Signals for identifying problem types |
| OTHERS | `oth` | Miscellaneous insights |

## API Reference

### ACEMiddleware

```python
ACEMiddleware(
    reflector_model: str | BaseChatModel | None = None,
    curator_model: str | BaseChatModel | None = None,
    initial_playbook: str | None = None,
    curator_frequency: int = 5,
    playbook_token_budget: int = 80000,
    enable_reflection: bool = True,
    enable_curation: bool = True,
    auto_prune: bool = False,
    prune_threshold: float = 0.5,
    prune_min_interactions: int = 3,
)
```

### Playbook Utilities

```python
from langchain.agents.middleware.ace import (
    ACEPlaybook,              # Dataclass for playbook state
    initialize_empty_playbook,  # Create default playbook template
    parse_playbook_line,      # Parse a bullet line
    format_playbook_line,     # Format a bullet line
    extract_bullet_ids,       # Extract [xxx-00000] IDs from text
    update_bullet_counts,     # Update helpful/harmful counts
    get_playbook_stats,       # Get playbook statistics
)
```

## References

- **Paper**: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)
- **Original Implementation**: [github.com/ace-agent/ace](https://github.com/ace-agent/ace)

