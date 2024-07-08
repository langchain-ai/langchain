from typing import Any, List


class MockEncoder:
    def encode(self, to_encode: str) -> str:
        return "[encoded]" + to_encode


class MockEncoderReturnsList:
    def encode(self, to_encode: Any) -> List:
        if isinstance(to_encode, str):
            return [1.0, 2.0]
        elif isinstance(to_encode, List):
            return [[1.0, 2.0] for _ in range(len(to_encode))]
        raise ValueError("Invalid input type for unit test")
