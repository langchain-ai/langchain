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

The reflector is currently "unsupervised" in that is does not use ground truth for training. Instead, it learns onelinu using the following information:

Did tools error? → Harmful
Did reasoning look sound? → Helpful
Did the agent seem confident? → Neutral/Helpful

We can certainly change the agent to be trainable to use gold labels and/or user feedback.

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

### 2. Ground Truth Comparison

**Status**: Not implemented

**Description**: The ability to compare agent responses against known correct answers
for training/evaluation scenarios.

**Why it matters**: The original ACE framework uses ground truth to provide precise
error analysis for training.

**Potential implementation**:
```python
# Option A: Evaluation mode with ground truth
ace = ACEMiddleware(
    ground_truth_provider=lambda question: lookup_answer(question)
)

# Option B: State-based ground truth
agent.invoke({
    "messages": [...],
    "ace_ground_truth": "The correct answer is 36"
})

# Option C: Batch evaluation mode
results = ace.evaluate(
    questions=["What is 15% of 240?", ...],
    ground_truth=["36", ...],
)
```

**Use cases**:
- Training on benchmark datasets
- A/B testing playbook variations
- Automated regression testing

---

## Impact on Reflection Quality

| Feedback Source | Precision | When Available |
|-----------------|-----------|----------------|
| Ground Truth | Very High | Training/eval only |
| User Feedback | High | Requires user action |
| Tool Results | Medium | When tools are used |
| Reasoning Analysis | Low | Always available |

The current implementation relies primarily on tool results and reasoning analysis,
which provides moderate reflection quality but may miss subtle errors that would be
caught with ground truth or user feedback.

## Future Enhancements

- **BulletpointAnalyzer**: Semantic deduplication of similar playbook bullets using
  embeddings. The original ACE uses this to prevent playbook bloat by merging
  semantically similar strategies. Could be added as an optional post-curation step.
