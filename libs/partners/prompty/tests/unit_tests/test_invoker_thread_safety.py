"""Test thread-safety of InvokerFactory singleton pattern."""

import threading

from langchain_prompty.core import InvokerFactory


def test_invoker_factory_singleton_thread_safety() -> None:
    """Test that InvokerFactory maintains singleton pattern under concurrent access.

    This test verifies the fix for issue #34981 by ensuring that multiple threads
    accessing InvokerFactory simultaneously all receive the same instance.
    """
    # Reset singleton for testing
    InvokerFactory._instance = None

    results: list[int] = []

    def thread_worker() -> None:
        """Worker function that gets InvokerFactory instance."""
        factory = InvokerFactory()
        results.append(id(factory))

    # Create 10 threads to access the singleton concurrently
    threads = [threading.Thread(target=thread_worker) for _ in range(10)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify all threads received the same instance
    unique_instances = set(results)
    assert len(unique_instances) == 1, (
        f"Expected 1 unique instance, but got {len(unique_instances)}. "
        f"Race condition detected!"
    )


def test_invoker_factory_singleton_consistency() -> None:
    """Test that multiple sequential calls return the same instance."""
    factory1 = InvokerFactory()
    factory2 = InvokerFactory()
    factory3 = InvokerFactory()

    assert factory1 is factory2
    assert factory2 is factory3
    assert id(factory1) == id(factory2) == id(factory3)


def test_invoker_factory_noop_registrations() -> None:
    """Test that NOOP invokers are properly registered."""
    factory = InvokerFactory()

    # Verify NOOP invokers are registered
    assert "NOOP" in factory._renderers
    assert "NOOP" in factory._parsers
    assert "NOOP" in factory._executors
    assert "NOOP" in factory._processors
