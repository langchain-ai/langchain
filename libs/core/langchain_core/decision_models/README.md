# Beta decision models

`BaseDecisionModel` provides a shared Runnable interface for bounded judgments:

- `Noul`: the probability that a statement is true, without thresholding.
- `Choice`: one of the supplied alternatives, with a complete distribution.
- `Score`: a probability-weighted position along ordered, zero-based rubric levels.

This interface is in beta. These names follow an emerging cross-provider convention;
they are not an industry standard. Providers can have different capabilities and
confidence formulas. OpenAI's native `Predicate` maps to canonical `Noul`.

## One request, interchangeable implementations

```python
from langchain_core.decision_models import (
    Choice,
    DecisionLevel,
    DecisionOption,
    DecisionRequest,
    Noul,
    Score,
)

request: DecisionRequest = {
    "state": "Payments have failed for three days. Please help!",
    "questions": {
        "urgent": Noul(instructions="Is this urgent?"),
        "team": Choice(
            instructions="Which team?",
            options=[
                DecisionOption(value="billing"),
                DecisionOption(value="technical"),
            ],
        ),
        "severity": Score(
            instructions="How severe?",
            levels=[
                DecisionLevel(label="calm"),
                DecisionLevel(label="frustrated"),
                DecisionLevel(label="angry"),
            ],
        ),
    },
}

# Use already-configured native instances, retaining their clients and SDK policies.
openai_model = openai_decisions.as_decision_model()
jev_model = typesafe_classifier.as_decision_model()
openai_result = openai_model.invoke(request)
jev_result = jev_model.invoke(request)

answer = openai_result.answers["urgent"]
if answer.type == "noul":
    probability = answer.probability  # Application code chooses any action threshold.
```

The additive adapter methods require a core release containing `decision_models`.
Existing native request/result types, serializers, credentials, and defaults retain
their behavior. Private adapters retain the configured native instance; they do not
close caller-owned clients and do not support LangChain model deserialization.

Equivalent validated question dictionaries are accepted at invocation. Questions
share state and are independent; use a later invocation for a dependent judgment.
Boolean choice values retain their identity through JSON serialization. The TypeSafe
adapter rejects boolean options and message media before making a network request.
The OpenAI adapter accepts text and embedded images and rejects unsupported media.
Known unsupported capabilities raise `ModelInvalidRequestError`; missing profile
information is unknown, not a promise of support.

## Results and validation

Every question must have exactly one matching answer or an explicit `RefusalAnswer`.
The base checks option identity, complete distributions, finite bounded probabilities,
and agreement between scores and their supplied rubric. Probability sums and score
means use an absolute rounding tolerance of `1e-3`; invalid distributions are never
silently renormalized. Provider adapters reject duplicate wire IDs/options before
converting arrays to mappings.

`NoulAnswer.probability` is the probability of truth. `ChoiceAnswer.selected_probability`
is the selected alternative's probability. Optional `provider_confidence` retains the
native confidence measure; neither field guarantees calibrated domain accuracy.
Provider-specific diagnostics and reported abstention are retained separately.
Refusal, abstention, a caller-defined `cannot_tell` option, transport failure, and an
application action are distinct.

Usage counts remain `None` when absent. A total exists only when both components are
reported. Resolved model identity, gateway IDs, cost, raw usage details, and recognized
request-ID headers are preserved when provided. Tracing uses existing LLM conventions
and records only validated, known counts; it does not synthesize chat messages.

## Execution and extension

The base inherits Runnable composition, configuration, batching, retries, and
fallbacks. `stream` and `astream` yield one complete response. An exception fallback
does not automatically run for a returned refusal. SDK retry policy stays with the
provider; configure Runnable retry exception types explicitly because
`ModelError.is_retryable` does not itself control `with_retry`.

Implement `_decide(request, *, config, run_manager)` to return a canonical
`DecisionResponse`. Optionally implement `_adecide` for native async transport. The
default executor fallback preserves context but cannot cancel an already-running
synchronous transport. Invocation keyword arguments are rejected; configure stable
provider settings on the model or through Runnable configurable fields.

`FakeDecisionModel(response=...)` exercises the same validation and callbacks offline,
without credentials or a provider SDK. Reusable offline and opt-in live conformance
suites are available in `langchain_tests.unit_tests.decision_models` and
`langchain_tests.integration_tests.decision_models`. Unit suites must inject mocks.
Consumer code can accept `Runnable[DecisionRequest, DecisionResponse]`; nominal
inheritance is not required for a compatible consumer seam.

Model caching, automatic action policies, partial streaming, provider discovery,
additional transports, and a second `Predicate` alias are deferred.
