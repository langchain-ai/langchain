# ACE Middleware

**ACE (Agentic Context Engineering)** is a middleware that enables LangChain agents to self-improve by maintaining an evolving "playbook" of strategies, insights, and patterns learned from interactions.

Based on the research paper: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)
and [original implementation](https://github.com/ace-agent/ace)

(Note: it is not a "one-to-one mapping" of components and architecture in the original
implementation to the implementation in LangGraph. See [Design Tradeoffs](Limitations.md) for details.)

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Design](#design)
  - [Three-Role Architecture](#three-role-architecture)
  - [The Playbook](#the-playbook)
  - [Middleware Hooks](#middleware-hooks)
- [How Trajectories Update the Playbook](#how-trajectories-update-the-playbook)
- [Usage Example](#usage-example)
- [State Schema](#state-schema)
- [Playbook Sections](#playbook-sections)
- [API Reference](#api-reference)
- [References](#references)
- [Appendix](#appendix)
  - [Why Helpful/Harmful Counts Improve Accuracy](#why-helpfulharmful-counts-improve-accuracy)
  - [How the Reflector Determines Helpful/Harmful](#how-the-reflector-determines-helpfulharmful)

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

## Quick Start

Run the demo to see ACE in action:

```bash
cd libs/langchain_v1/langchain/agents/middleware/ace
uv run --env-file .env python examples/ace_playbook_demo.py
```

This shows the playbook evolving as the agent solves math problems, with the reflector
analyzing each response and the curator adding new insights.

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
## strategies_and_insights
[str-00001] helpful=5 harmful=0 :: Always verify data types before processing
[str-00002] helpful=3 harmful=1 :: Consider edge cases in financial data

## formulas_and_calculations
[cal-00001] helpful=8 harmful=0 :: NPV = Σ(Cash Flow / (1+r)^t)

## common_mistakes_to_avoid
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

### 5. Automatic Pruning (TODO and Optional)

When `auto_prune=True`, bullets with high harmful-to-helpful ratios are removed:

```python
# Bullet with harmful > helpful gets pruned
[str-00005] helpful=1 harmful=5 :: Bad advice  # REMOVED
```


## Usage Example

### Basic Usage

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ACEMiddleware
from langchain_core.messages import HumanMessage

# Create ACE middleware (both models are required)
ace = ACEMiddleware(
    reflector_model="gpt-4o-mini",  # Analyzes trajectories
    curator_model="gpt-4o-mini",    # Curates the playbook
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
    initial_playbook="""## strategies_and_insights
[str-00001] helpful=10 harmful=0 :: Always verify data types before processing
[str-00002] helpful=5 harmful=0 :: Break complex problems into smaller steps

## common_mistakes_to_avoid
[mis-00001] helpful=8 harmful=0 :: Don't forget to handle timezone conversions
[mis-00002] helpful=6 harmful=0 :: Validate user inputs before processing
""",
)
```

### Full Configuration

```python
ace = ACEMiddleware(
    # Models for reflection and curation (both required)
    reflector_model="gpt-4o-mini",      # Or pass a BaseChatModel instance
    curator_model="gpt-4o-mini",        # Or pass a BaseChatModel instance

    # Initial playbook content
    initial_playbook=None,              # Uses default template if not provided

    # Curation settings
    curator_frequency=5,                # Run curator every N interactions
    playbook_token_budget=80000,        # Max tokens for playbook

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

The default playbook template includes these sections (using normalized snake_case names):

| Section | Slug | Purpose |
|---------|------|---------|
| `strategies_and_insights` | `str` | General approaches and tactics |
| `formulas_and_calculations` | `cal` | Mathematical formulas and computation patterns |
| `code_snippets_and_templates` | `cod` | Reusable code patterns |
| `common_mistakes_to_avoid` | `mis` | Known pitfalls and anti-patterns |
| `problem_solving_heuristics` | `heu` | Decision-making rules of thumb |
| `context_clues_and_indicators` | `ctx` | Signals for identifying problem types |
| `others` | `oth` | Miscellaneous insights |

## API Reference

### ACEMiddleware

```python
ACEMiddleware(
    reflector_model: str | BaseChatModel,      # Required
    curator_model: str | BaseChatModel,        # Required
    initial_playbook: str | None = None,
    curator_frequency: int = 5,
    playbook_token_budget: int = 80000,
    auto_prune: bool = False,
    prune_threshold: float = 0.5,
    prune_min_interactions: int = 3,
    expected_interactions: int | None = None,
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
---

## References

- **Paper**: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)
- **Original Implementation**: [github.com/ace-agent/ace](https://github.com/ace-agent/ace)

## Appendix

## Why Helpful/Harmful Counts Improve Accuracy

The core insight of ACE: instead of retraining models, accumulate and refine contextual
knowledge that gets injected into prompts. The count system makes this work.

### Count Semantics

| Tag | Meaning | Count Update |
|-----|---------|--------------|
| `helpful` | Bullet contributed to correct response | `helpful += 1` |
| `harmful` | Bullet led to errors or wrong approach | `harmful += 1` |
| `neutral` | Bullet was used but had no significant impact | No change |

### Evolutionary Optimization

The playbook evolves like a genetic algorithm:

- **Selection**: High helpful/harmful ratio bullets survive
- **Mutation**: Curator adds new strategies based on reflections
- **Extinction**: Harmful bullets get pruned

Over many interactions, the playbook converges on strategies that actually work.

### Signal Visibility

Counts are visible in the playbook, so the agent can see reliability:

```
[str-00001] helpful=50 harmful=1 :: Proven reliable strategy
[str-00002] helpful=2 harmful=8 :: Risky approach (will be pruned)
```

### Noise Tolerance

Unlike binary keep/delete, counts provide:

- **Confidence**: `helpful=50 harmful=2` is very reliable
- **Noise tolerance**: One bad outcome doesn't kill a good strategy
- **Trend detection**: Rising harmful count signals degrading usefulness

### Playbook Statistics

The system tracks aggregate stats for the curator:

```python
stats = {
    'total_bullets': 25,
    'high_performing': 8,   # helpful > 5, harmful < 2
    'problematic': 3,       # harmful >= helpful
    'unused': 5,            # helpful + harmful = 0
}
```

## How the Reflector Determines Helpful/Harmful

The reflector is an LLM that analyzes the agent's response and makes inferences about
which playbook bullets helped or hurt. Here's what it receives:

### Input to the Reflector

```
**User Query:** "What is 15% of 240?"

**Agent's Reasoning/Response:** "Using the calculator, 15 * 240 = 3600..."

**Feedback/Outcome:**
- Called 'calculator' with: {'expression': '15 * 240'}
- Tool 'calculator': 3600

**Playbook Bullets Referenced:**
[str-00001] helpful=0 harmful=0 :: Break word problems into steps
[mis-00001] helpful=0 harmful=0 :: Convert percentages to decimals
```

### How Tagging Works

The reflector LLM reasons about:

1. **Did the agent follow the bullet's advice?**
   - If `[mis-00001]` says "convert percentages" but agent used `15 * 240` → **harmful**

2. **Did following the advice lead to success?**
   - If agent followed `[str-00001]` and got correct answer → **helpful**

3. **Was the advice irrelevant to the outcome?**
   - If bullet was used but had no bearing → **neutral**

### Feedback Sources

The reflector receives feedback from:

| Source | Signal Quality | When Available |
|--------|---------------|----------------|
| Tool errors | High | When tools fail |
| Tool results | Medium | When tools succeed |
| Reasoning analysis | Low | Always |

### Current Limitations

Without ground truth (the correct answer), the reflector must **infer** correctness based on:
- Tool results and errors (extracted from message history)
- Reasoning quality analysis
- LLM's domain knowledge

This is less accurate than the original ACE which compares `predicted_answer` vs `ground_truth`.
See [Limitations.md](Limitations.md) for details.


