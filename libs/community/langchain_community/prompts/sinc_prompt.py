"""SincPromptTemplate for LangChain.

Implements the sinc-prompt format for structured LLM prompts based on
the Nyquist-Shannon sampling theorem.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Specification: https://tokencalc.pro/spec
JSON Schema: https://tokencalc.pro/schema/sinc-prompt-v1.json
Paper: https://doi.org/10.5281/zenodo.19152668
PyPI: pip install sinc-llm
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import Field, model_validator


# The 6 mandatory specification bands
SINC_BANDS = ["PERSONA", "CONTEXT", "DATA", "CONSTRAINTS", "FORMAT", "TASK"]

# Importance weights from MATLAB dropout simulation (275 observations)
BAND_IMPORTANCE = {
    "PERSONA": 0.070,
    "CONTEXT": 0.063,
    "DATA": 0.038,
    "CONSTRAINTS": 0.427,
    "FORMAT": 0.263,
    "TASK": 0.028,
}


class SincPromptTemplate(BasePromptTemplate):
    """A prompt template based on the sinc-prompt specification.

    The sinc-prompt format decomposes prompts into 6 frequency bands on
    the specification axis, following the Nyquist-Shannon sampling theorem.
    Each band carries a measured importance weight for output quality.

    The 6 bands are:
        n=0 PERSONA (7.0%): Who should answer
        n=1 CONTEXT (6.3%): Situation and background
        n=2 DATA (3.8%): Specific inputs and metrics
        n=3 CONSTRAINTS (42.7%): Behavioral rules (highest importance)
        n=4 FORMAT (26.3%): Output structure
        n=5 TASK (2.8%): The specific objective

    Example:
        .. code-block:: python

            from langchain_community.prompts import SincPromptTemplate

            template = SincPromptTemplate(
                persona="You are a senior Python developer.",
                context="Production codebase with 50K users.",
                data="Error rate: 2.3%. Affected endpoint: /api/users.",
                constraints="Never hedge. Use exact line numbers. No refactoring beyond the fix.",
                format="Bug (1 sentence), corrected code block, test output.",
                task="Find and fix the authentication bug.",
            )

            prompt = template.format()

    Reference:
        - Spec: https://tokencalc.pro/spec
        - Paper: https://doi.org/10.5281/zenodo.19152668
        - GitHub: https://github.com/mdalexandre/sinc-llm
    """

    persona: str = Field(default="", description="n=0: Who should answer (7.0% importance)")
    context: str = Field(default="", description="n=1: Situation and background (6.3%)")
    data: str = Field(default="", description="n=2: Specific inputs and metrics (3.8%)")
    constraints: str = Field(default="", description="n=3: Behavioral rules (42.7% importance)")
    fmt: str = Field(default="", description="n=4: Output structure (26.3%)")
    task: str = Field(default="", description="n=5: The specific objective (2.8%)")

    input_variables: List[str] = Field(default_factory=list)

    @classmethod
    def from_sinc_json(cls, sinc_json: Union[str, Path, Dict]) -> "SincPromptTemplate":
        """Create a SincPromptTemplate from sinc-format JSON.

        Args:
            sinc_json: A sinc JSON dict, JSON string, or path to a .sinc.json file.

        Returns:
            SincPromptTemplate with all 6 bands populated.
        """
        if isinstance(sinc_json, (str, Path)):
            path = Path(sinc_json)
            if path.exists():
                sinc_json = json.loads(path.read_text(encoding="utf-8"))
            else:
                sinc_json = json.loads(str(sinc_json))

        fragments = {f["t"]: f["x"] for f in sinc_json.get("fragments", [])}

        return cls(
            persona=fragments.get("PERSONA", ""),
            context=fragments.get("CONTEXT", ""),
            data=fragments.get("DATA", ""),
            constraints=fragments.get("CONSTRAINTS", ""),
            fmt=fragments.get("FORMAT", ""),
            task=fragments.get("TASK", ""),
        )

    def format(self, **kwargs: Any) -> str:
        """Format the sinc prompt into a string."""
        parts = []
        bands = [
            ("PERSONA", self.persona),
            ("CONTEXT", self.context),
            ("DATA", self.data),
            ("CONSTRAINTS", self.constraints),
            ("FORMAT", self.fmt),
            ("TASK", self.task),
        ]

        for name, content in bands:
            if content:
                parts.append(f"[{name}]\n{content}")

        return "\n\n".join(parts)

    async def aformat(self, **kwargs: Any) -> str:
        """Async format."""
        return self.format(**kwargs)

    def format_prompt(self, **kwargs: Any) -> str:
        """Format prompt."""
        return self.format(**kwargs)

    def to_sinc_json(self) -> Dict[str, Any]:
        """Export as sinc-format JSON."""
        fragments = []
        bands = [
            (0, "PERSONA", self.persona),
            (1, "CONTEXT", self.context),
            (2, "DATA", self.data),
            (3, "CONSTRAINTS", self.constraints),
            (4, "FORMAT", self.fmt),
            (5, "TASK", self.task),
        ]

        for n, t, x in bands:
            fragments.append({"n": n, "t": t, "x": x})

        return {
            "formula": "x(t) = \u03a3 x(nT) \u00b7 sinc((t - nT) / T)",
            "T": "specification-axis",
            "fragments": fragments,
        }

    def nyquist_completeness(self) -> float:
        """Compute Nyquist completeness score (0.0 to 1.0).

        Returns the weighted fraction of bands that are populated,
        using the MATLAB-derived importance weights.
        """
        total = 0.0
        present = 0.0
        bands = {
            "PERSONA": self.persona,
            "CONTEXT": self.context,
            "DATA": self.data,
            "CONSTRAINTS": self.constraints,
            "FORMAT": self.fmt,
            "TASK": self.task,
        }

        for name, content in bands.items():
            weight = BAND_IMPORTANCE[name]
            total += weight
            if content.strip():
                present += weight

        return round(present / total, 4) if total > 0 else 0.0

    @property
    def _prompt_type(self) -> str:
        return "sinc-prompt"
