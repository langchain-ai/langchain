"""Comparison test demonstrating the improvement from list.pop(0) to deque.popleft()."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def process_with_list(items: list[str]) -> None:
    """Process items using list.pop(0) - O(n) per operation."""
    while items:
        _ = items.pop(0)


def process_with_deque(items: deque[str]) -> None:
    """Process items using deque.popleft() - O(1) per operation."""
    while items:
        _ = items.popleft()


def benchmark_comparison() -> None:
    """Compare performance between list.pop(0) and deque.popleft()."""
    print("\n" + "=" * 70)
    print("Complexity Comparison: list.pop(0) vs deque.popleft()")
    print("=" * 70 + "\n")

    test_sizes = [100, 500, 1000, 5000, 10000]

    print(f"{'Size':<10} {'List.pop(0)':<20} {'Deque.popleft()':<20} {'Speedup':<15}")
    print("-" * 70)

    for size in test_sizes:
        # Test with list.pop(0)
        test_list = list(range(size))
        start = time.perf_counter()
        process_with_list(test_list)
        list_time = time.perf_counter() - start

        # Test with deque.popleft()
        test_deque = deque(range(size))
        start = time.perf_counter()
        process_with_deque(test_deque)
        deque_time = time.perf_counter() - start

        speedup = list_time / deque_time if deque_time > 0 else 0

        print(
            f"{size:<10} {list_time*1000:<20.4f}ms {deque_time*1000:<20.4f}ms {speedup:<15.2f}x"
        )

    print("\n" + "=" * 70)
    print("Note: The speedup increases with size due to O(n) vs O(1) complexity")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    benchmark_comparison()
