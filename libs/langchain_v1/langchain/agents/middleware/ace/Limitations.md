# ACE Middleware Limitations

This document outlines current limitations of the LangChain ACE middleware compared to
the full ACE framework, along with potential future enhancements.

## Current Feedback Sources

The reflector currently receives feedback from:

- **Tool Results**: ✅ Implemented - Tool call results and errors are extracted from
  the message history and passed to the reflector for analysis.

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
error analysis. Without it, the reflector is essentially "guessing" what went wrong
based on reasoning quality alone.

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

## Recommendations

1. **For production agents**: Implement user feedback collection to improve
   reflection accuracy over time.

2. **For training/fine-tuning**: Add ground truth support to enable precise
   error analysis and faster playbook convergence.

3. **For debugging**: Tool feedback is already sufficient to identify most
   execution errors and failed strategies.

