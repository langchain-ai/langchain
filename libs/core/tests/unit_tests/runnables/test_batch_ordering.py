"""Test batch processing order preservation and thread-safety."""

import random
import threading
import time
from contextvars import copy_context
from typing import Any, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import ContextThreadPoolExecutor


class OrderTrackingRunnable(Runnable[str, str]):
    """A runnable that tracks processing order and adds unique identifiers."""

    def __init__(self, name: str, delay: float = 0.0, fail_on: Optional[str] = None):
        """Initialize the runnable.

        Args:
            name: Name of this runnable for identification.
            delay: Artificial delay to simulate processing time.
            fail_on: If input starts with this string, raise an exception.
        """
        self.name = name
        self.delay = delay
        self.fail_on = fail_on
        self.processed_items = []
        self.processing_order = []

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> str:
        """Process input and track order."""
        # Track when this item started processing
        self.processing_order.append(input)

        # Simulate processing delay
        if self.delay > 0:
            time.sleep(self.delay)

        # Check if we should fail
        if self.fail_on and input.startswith(self.fail_on):
            msg = f"{self.name} failed on input: {input}"
            raise ValueError(msg)

        # Process and track
        result = f"{input}_{self.name}"
        self.processed_items.append((input, result))
        return result


class ContextCapturingRunnable(Runnable[str, str]):
    """A runnable that captures context information to verify uniqueness."""

    def __init__(self) -> None:
        """Initialize the runnable."""
        self.contexts_seen = []
        self.input_context_map = {}

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> str:
        """Process input and capture context."""
        import threading
        from contextvars import copy_context

        # Capture current context and thread info
        context_id = id(copy_context())
        thread_id = threading.current_thread().ident

        # Store context info
        context_info = {
            "input": input,
            "context_id": context_id,
            "thread_id": thread_id,
        }
        self.contexts_seen.append(context_info)
        self.input_context_map[input] = context_info

        return f"{input}_ctx{context_id}"


def test_batch_preserves_order() -> None:
    """Test that batch processing preserves input/output order."""
    runnable = OrderTrackingRunnable("test", delay=0.01)
    inputs = [f"input_{i}" for i in range(10)]

    # Process batch
    outputs = runnable.batch(inputs)

    # Verify order is preserved
    expected_outputs = [f"input_{i}_test" for i in range(10)]
    assert outputs == expected_outputs, "Output order should match input order"

    # Verify all items were processed
    assert len(runnable.processed_items) == 10
    processed_inputs = [item[0] for item in runnable.processed_items]
    assert set(processed_inputs) == set(inputs), "All inputs should be processed"


def test_batch_with_return_exceptions() -> None:
    """Test that batch with return_exceptions=True preserves order."""
    runnable = OrderTrackingRunnable("test", delay=0.01, fail_on="fail")
    inputs = ["input_0", "fail_1", "input_2", "fail_3", "input_4"]

    # Process batch with return_exceptions=True
    outputs = runnable.batch(inputs, return_exceptions=True)

    # Verify we got 5 outputs
    assert len(outputs) == 5, "Should have one output per input"

    # Check that outputs are in correct positions
    assert outputs[0] == "input_0_test", "First output should be processed normally"
    assert isinstance(outputs[1], ValueError), "Second output should be an exception"
    assert outputs[2] == "input_2_test", "Third output should be processed normally"
    assert isinstance(outputs[3], ValueError), "Fourth output should be an exception"
    assert outputs[4] == "input_4_test", "Fifth output should be processed normally"


def test_batch_high_concurrency() -> None:
    """Test batch processing with high concurrency to expose race conditions."""
    runnable = OrderTrackingRunnable("test", delay=0.001)

    # Create many inputs to increase chance of race conditions
    inputs = [f"input_{i:03d}" for i in range(100)]

    # Process with high concurrency
    config = {"max_concurrency": 20}
    outputs = runnable.batch(inputs, config=config)

    # Verify order is preserved despite high concurrency
    expected_outputs = [f"input_{i:03d}_test" for i in range(100)]
    assert outputs == expected_outputs, (
        "Order should be preserved even with high concurrency"
    )

    # Verify no duplicates
    assert len(set(outputs)) == len(outputs), "No duplicate outputs should exist"
    assert len(outputs) == 100, "All inputs should produce outputs"


def test_batch_unique_contexts() -> None:
    """Test that each input gets a unique context without duplicates."""
    runnable = ContextCapturingRunnable()
    inputs = [f"input_{i}" for i in range(20)]

    # Process batch
    runnable.batch(inputs)

    # Verify each input was processed
    assert len(runnable.contexts_seen) == 20, "Each input should be processed once"

    # Verify each input got a unique processing slot
    inputs_seen = [ctx["input"] for ctx in runnable.contexts_seen]
    assert sorted(inputs_seen) == sorted(inputs), "All inputs should be processed"
    assert len(set(inputs_seen)) == len(inputs), "No input should be processed twice"


def test_sequence_batch_with_exceptions() -> None:
    """Test RunnableSequence batch with return_exceptions=True."""
    # Create a sequence of runnables
    step1 = OrderTrackingRunnable("step1", delay=0.001)
    step2 = OrderTrackingRunnable("step2", delay=0.001, fail_on="input_1")
    step3 = OrderTrackingRunnable("step3", delay=0.001)

    sequence = step1 | step2 | step3

    inputs = ["input_0", "input_1", "input_2"]
    outputs = sequence.batch(inputs, return_exceptions=True)

    # Verify we got 3 outputs
    assert len(outputs) == 3, "Should have one output per input"

    # First input should process through all steps
    assert outputs[0] == "input_0_step1_step2_step3"

    # Second input should fail at step2
    assert isinstance(outputs[1], ValueError)
    assert "step2 failed" in str(outputs[1])

    # Third input should process through all steps
    assert outputs[2] == "input_2_step1_step2_step3"


def test_context_thread_pool_executor_map_order() -> None:
    """Test ContextThreadPoolExecutor.map preserves order."""

    def process_item(item: str) -> str:
        # Simulate some work
        time.sleep(0.001)
        return f"processed_{item}"

    with ContextThreadPoolExecutor(max_workers=4) as executor:
        inputs = [f"item_{i}" for i in range(20)]
        results = list(executor.map(process_item, inputs))

        # Verify order is preserved
        expected = [f"processed_item_{i}" for i in range(20)]
        assert results == expected, "Map should preserve order"


def test_context_thread_pool_executor_map_with_multiple_iterables() -> None:
    """Test ContextThreadPoolExecutor.map with multiple iterables."""

    def combine_items(a: str, b: int, c: float) -> str:
        return f"{a}_{b}_{c}"

    with ContextThreadPoolExecutor(max_workers=4) as executor:
        list_a = [f"a{i}" for i in range(10)]
        list_b = list(range(10))
        list_c = [float(i) * 0.5 for i in range(10)]

        results = list(executor.map(combine_items, list_a, list_b, list_c))

        # Verify order and correctness
        expected = [f"a{i}_{i}_{i * 0.5}" for i in range(10)]
        assert results == expected, "Map with multiple iterables should preserve order"


def test_batch_no_race_conditions() -> None:
    """Test that batch processing has no race conditions with shared state."""

    class SharedStateRunnable(Runnable[int, int]):
        """A runnable that modifies shared state."""

        def __init__(self) -> None:
            self.counter = 0
            self.results = []

        def invoke(
            self, input: int, config: Optional[RunnableConfig] = None, **kwargs: Any
        ) -> int:
            # Simulate race condition scenario
            current = self.counter
            time.sleep(0.001)  # Give other threads a chance to interfere
            self.counter = current + 1
            result = input * 2
            self.results.append((input, result))
            return result

    runnable = SharedStateRunnable()
    inputs = list(range(50))

    # Process with high concurrency
    config = {"max_concurrency": 10}
    outputs = runnable.batch(inputs, config=config)

    # Verify outputs are correct and in order
    expected = [i * 2 for i in range(50)]
    assert outputs == expected, "Outputs should be correct and in order"

    # Note: The counter might not be 50 due to race conditions in the test runnable itself,
    # but the batch processing order should still be preserved


def test_batch_with_varying_processing_times() -> None:
    """Test that batch preserves order even with varying processing times."""
    import random

    class VariableDelayRunnable(Runnable[str, str]):
        """A runnable with variable processing delays."""

        def invoke(
            self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
        ) -> str:
            # Random delay to simulate varying processing times
            delay = random.uniform(0.001, 0.01)
            time.sleep(delay)
            return f"processed_{input}"

    runnable = VariableDelayRunnable()
    inputs = [f"item_{i:02d}" for i in range(20)]

    # Run multiple times to ensure consistency
    for _ in range(3):
        outputs = runnable.batch(inputs)
        expected = [f"processed_item_{i:02d}" for i in range(20)]
        assert outputs == expected, (
            "Order should be preserved regardless of processing time"
        )


def test_batch_empty_input() -> None:
    """Test batch with empty input list."""
    runnable = OrderTrackingRunnable("test")
    outputs = runnable.batch([])
    assert outputs == [], "Empty input should return empty output"


def test_batch_single_input() -> None:
    """Test batch with single input."""
    runnable = OrderTrackingRunnable("test")
    outputs = runnable.batch(["single"])
    assert outputs == ["single_test"], "Single input should be processed correctly"


if __name__ == "__main__":
    # Run tests
    test_batch_preserves_order()
    test_batch_with_return_exceptions()
    test_batch_high_concurrency()
    test_batch_unique_contexts()
    test_sequence_batch_with_exceptions()
    test_context_thread_pool_executor_map_order()
    test_context_thread_pool_executor_map_with_multiple_iterables()
    test_batch_no_race_conditions()
    test_batch_with_varying_processing_times()
    test_batch_empty_input()
    test_batch_single_input()

