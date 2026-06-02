#!/usr/bin/env python
"""Simple verification script for the Pydantic 2.14+ fix."""

import sys
from typing import Any

from pydantic import BaseModel

from langchain_core.utils.pydantic import _create_subset_model_v2


def test_annotations_invariant():
    """Test that __annotations__ follows Python's invariant."""
    print("Testing __annotations__ invariant...")

    class TestModel(BaseModel):
        field1: str
        field2: int
        field3: dict[str, Any]

    # Create subset model
    subset = _create_subset_model_v2("TestSubset", TestModel, ["field1", "field3"])

    # Check the Python invariant
    getattr_result = getattr(subset, "__annotations__", {})
    dict_result = subset.__dict__.get("__annotations__", {})

    if getattr_result != dict_result:
        print(f"❌ FAILED: Invariant violated")
        print(f"   getattr result: {getattr_result}")
        print(f"   __dict__ result: {dict_result}")
        return False

    # Verify selected fields are in annotations
    if "field1" not in subset.__annotations__ or "field3" not in subset.__annotations__:
        print(f"❌ FAILED: Missing expected fields in annotations")
        return False

    if "field2" in subset.__annotations__:
        print(f"❌ FAILED: Excluded field found in annotations")
        return False

    print("✅ PASSED: __annotations__ invariant preserved")
    return True


def test_complex_annotations():
    """Test with complex type annotations."""
    print("Testing complex annotations...")

    class TestModel(BaseModel):
        list_field: list[dict[str, Any]]
        union_field: str | int | None
        simple_field: str

    subset = _create_subset_model_v2(
        "TestSubset", TestModel, ["list_field", "union_field"]
    )

    # Verify invariant
    if getattr(subset, "__annotations__", {}) != subset.__dict__.get("__annotations__", {}):
        print(f"❌ FAILED: Invariant violated with complex types")
        return False

    # Verify fields
    if "list_field" not in subset.__annotations__ or "union_field" not in subset.__annotations__:
        print(f"❌ FAILED: Missing complex typed fields")
        return False

    if "simple_field" in subset.__annotations__:
        print(f"❌ FAILED: Excluded field present in complex test")
        return False

    print("✅ PASSED: Complex annotations handled correctly")
    return True


def test_instantiation():
    """Test that the subset model can be instantiated."""
    print("Testing model instantiation...")

    class TestModel(BaseModel):
        field_with_default: str = "default_value"
        field_without_default: int

    subset = _create_subset_model_v2(
        "TestSubset",
        TestModel,
        ["field_with_default", "field_without_default"],
    )

    try:
        instance = subset(field_without_default=42)
        if instance.field_with_default != "default_value":
            print(f"❌ FAILED: Default value not set correctly")
            return False

        if instance.field_without_default != 42:
            print(f"❌ FAILED: Field value not set correctly")
            return False

        print("✅ PASSED: Model instantiation works correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: Exception during instantiation: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Pydantic 2.14+ Fix Verification Tests")
    print("=" * 60)
    print()

    tests = [
        test_annotations_invariant,
        test_complex_annotations,
        test_instantiation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ FAILED: {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
