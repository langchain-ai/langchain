import re
from typing import List

from langchain.document_loaders.parsers.language.code_segmenter import CodeSegmenter


class CobolSegmenter(CodeSegmenter):
    """Code segmenter for `COBOL`."""

    # Regex pattern for paragraph names
    PARAGRAPH_PATTERN = re.compile(r"^[A-Z0-9\-]+\.$", re.IGNORECASE)

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines = self.code.splitlines()

    def is_valid(self) -> bool:
        # Check divisions at the start of lines to reduce false positives
        divisions = ["IDENTIFICATION DIVISION", "DATA DIVISION", "PROCEDURE DIVISION"]
        return any(
            line.startswith(div) for line in self.source_lines for div in divisions
        )

    def _extract_code(self, start_idx: int, end_idx: int) -> str:
        return "\n".join(self.source_lines[start_idx:end_idx])

    def extract_functions_classes(self) -> List[str]:
        paragraphs = []
        start_idx = None

        for i, line in enumerate(self.source_lines):
            if self.PARAGRAPH_PATTERN.match(line.strip()):
                if start_idx is not None:
                    paragraphs.append(self._extract_code(start_idx, i))
                start_idx = i

        if start_idx is not None:
            paragraphs.append(self._extract_code(start_idx, len(self.source_lines)))

        return paragraphs

    def simplify_code(self) -> str:
        simplified_lines = self.source_lines[:]
        start_idx = None

        for i, line in enumerate(self.source_lines):
            if self.PARAGRAPH_PATTERN.match(line.strip()):
                if start_idx is not None:
                    # Use list slicing for optimization
                    simplified_lines[start_idx + 1 : i] = [None] * (i - start_idx - 1)
                start_idx = i

        if start_idx is not None:
            simplified_lines[start_idx + 1 :] = [None] * (
                len(self.source_lines) - start_idx - 1
            )

        return "\n".join(line for line in simplified_lines if line is not None)
