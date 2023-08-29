class MockEncoder:
    def encode(self, to_encode: str) -> str:
        return "[encoded]" + to_encode
