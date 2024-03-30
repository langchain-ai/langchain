import re
from typing import Callable, List

from langchain_community.document_loaders.parsers.language.code_segmenter import (
    CodeSegmenter,
)


class CobolSegmenter(CodeSegmenter):
    """Code segmenter for `COBOL`."""

    PARAGRAPH_PATTERN = re.compile(r"^[A-Z0-9\-]+(\s+.*)?\.$", re.IGNORECASE)
    DIVISION_PATTERN = re.compile(
        r"^\s*(IDENTIFICATION|DATA|PROCEDURE|ENVIRONMENT)\s+DIVISION.*$", re.IGNORECASE
    )
    SECTION_PATTERN = re.compile(r"^\s*[A-Z0-9\-]+\s+SECTION.$", re.IGNORECASE)

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines: List[str] = self.code.splitlines()

    def is_valid(self) -> bool:
        # Identify presence of any division to validate COBOL code
        return any(self.DIVISION_PATTERN.match(line) for line in self.source_lines)

    def _extract_code(self, start_idx: int, end_idx: int) -> str:
        return "\n".join(self.source_lines[start_idx:end_idx]).rstrip("\n")

    def _is_relevant_code(self, line: str) -> bool:
        """Check if a line is part of the procedure division or a relevant section."""
        if "PROCEDURE DIVISION" in line.upper():
            return True
        # Add additional conditions for relevant sections if needed
        return False

    def _process_lines(self, func: Callable) -> List[str]:
        """A generic function to process COBOL lines based on provided func."""
        elements: List[str] = []
        start_idx = None
        inside_relevant_section = False

        for i, line in enumerate(self.source_lines):
            if self._is_relevant_code(line):
                inside_relevant_section = True

            if inside_relevant_section and (
                self.PARAGRAPH_PATTERN.match(line.strip().split(" ")[0])
                or self.SECTION_PATTERN.match(line.strip())
            ):
                if start_idx is not None:
                    func(elements, start_idx, i)
                start_idx = i

        # Handle the last element if exists
        if start_idx is not None:
            func(elements, start_idx, len(self.source_lines))

        return elements

    def extract_functions_classes(self) -> List[str]:
        def extract_func(elements: List[str], start_idx: int, end_idx: int) -> None:
            elements.append(self._extract_code(start_idx, end_idx))

        return self._process_lines(extract_func)

    def simplify_code(self) -> str:
        simplified_lines: List[str] = []
        inside_relevant_section = False
        omitted_code_added = (
            False  # To track if "* OMITTED CODE *" has been added after the last header
        )

        for line in self.source_lines:
            is_header = (
                "PROCEDURE DIVISION" in line
                or "DATA DIVISION" in line
                or "IDENTIFICATION DIVISION" in line
                or self.PARAGRAPH_PATTERN.match(line.strip().split(" ")[0])
                or self.SECTION_PATTERN.match(line.strip())
            )

            if is_header:
                inside_relevant_section = True
                # Reset the flag since we're entering a new section/division or
                # paragraph
                omitted_code_added = False

            if inside_relevant_section:
                if is_header:
                    # Add header and reset the omitted code added flag
                    simplified_lines.append(line)
                elif not omitted_code_added:
                    # Add omitted code comment only if it hasn't been added directly
                    # after the last header
                    simplified_lines.append("* OMITTED CODE *")
                    omitted_code_added = True

        return "\n".join(simplified_lines)
