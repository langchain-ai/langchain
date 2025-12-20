# ACE Middleware Design Trade-offs and Limitations

This document outlines current design tradeoffs and limitations of the LangChain ACE middleware compared to the full ACE framework, along with potential future enhancements.

## Architectural Differences: Middleware vs Dedicated Generator

The original ACE uses a **dedicated generator** that completely controls
the agent's input/output format, while LangChain ACE uses **middleware that injects
the playbook into the system prompt**.

### Original ACE: Dedicated Generator

```python
GENERATOR_PROMPT = """You are an analysis expert...

Your output should be a json object:
- reasoning: your chain of thought
- bullet_ids: bulletpoints you used
- final_answer: your concise final answer

**Playbook:**
{playbook}
"""
```
The generator IS the agent - it replaces the entire agent with ACE-specific behavior.

### LangChain ACE: System Prompt Injection

```python
PLAYBOOK_INJECTION_TEMPLATE = """{original_prompt}

---

## ACE PLAYBOOK
{playbook}

**IMPORTANT**: Include bullet IDs in: <!-- bullet_ids: [...] -->
"""
```

The middleware wraps an existing agent - it augments but doesn't replace.

### Tradeoff Comparison

| Aspect | Dedicated Generator | System Prompt Injection |
|--------|---------------------|------------------------|
| **Bullet Tracking** | ✅ Guaranteed (structured JSON) | ⚠️ Unreliable (agent may ignore) |
| **Composability** | ❌ Must use ACE's generator | ✅ Works with any agent/tools |
| **Tool Support** | ❌ Single-shot, no tools | ✅ Full tool support |
| **Multi-turn** | ❌ Designed for single Q&A | ✅ Works across conversation |
| **Integration Effort** | High (replace agent) | Low (add middleware) |
| **Best For** | Training/benchmarks | Production agents |

### Reliability of Bullet Tracking

The main limitation of the middleware approach is **unreliable bullet tracking**:

1. The agent is instructed to include `<!-- bullet_ids: [...] -->` comments
2. The agent may not follow this instruction consistently
3. Without bullet IDs, the reflector cannot accurately attribute helpful/harmful tags

**Potential mitigations** (not yet implemented):
- Use `with_structured_output()` on final responses
- Register a `report_bullets` tool the agent must call

---

## Current Feedback Sources

The reflector uses the following feedback signals:

| Source | Signal | When Available |
|--------|--------|----------------|
| Ground Truth | Compare agent answer to known correct answer | When `ground_truth` provided in state |
| Tool Results | Did tools error? → Harmful | When tools are used |
| Reasoning Analysis | Did reasoning look sound? → Helpful | Always available |

See [Training with Ground Truth](README.md#training-with-ground-truth) for usage.

## Missing Feedback Sources

### 1. User Feedback

**Status**: Not implemented

**Description**: The ability for users to provide explicit feedback on agent responses
(e.g., thumbs up/down, corrections, ratings).

**Why it matters**: User feedback provides the most reliable signal about response
quality. Without it, the reflector can only infer quality from tool results and
reasoning patterns.

**Potential implementation**:
```python
# Option A: Callback-based feedback
ace = ACEMiddleware(
    feedback_callback=lambda response: get_user_rating(response)
)

# Option B: State-based feedback
agent.invoke(
    {"messages": [...], "ace_user_feedback": "incorrect - forgot to convert units"}
)

# Option C: Post-hoc feedback API
ace.record_feedback(thread_id="...", rating="negative", correction="...")
```

**Challenges**:
- Async nature of user feedback (may come later)
- Mapping feedback to specific bullets
- Integrating with different UI frameworks

---

## Impact on Reflection Quality

| Feedback Source | Precision | When Available | Status |
|-----------------|-----------|----------------|--------|
| Ground Truth | Very High | When `ground_truth` in state | ✅ Implemented |
| User Feedback | High | Requires user action | Not implemented |
| Tool Results | Medium | When tools are used | ✅ Implemented |
| Reasoning Analysis | Low | Always available | ✅ Implemented |

With ground truth support, the reflector can now provide high-precision feedback
during training/evaluation scenarios. For production inference without ground truth,
the reflector relies on tool results and reasoning analysis.

## Future Enhancements

- **BulletpointAnalyzer**: Semantic deduplication of similar playbook bullets using
  embeddings. The original ACE uses this to prevent playbook bloat by merging
  semantically similar strategies. Could be added as an optional post-curation step.
