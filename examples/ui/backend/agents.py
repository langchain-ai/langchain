"""Load the four example agents once for reuse by the API layer.

Each agent is built from the canonical `build_agent()` in the corresponding
`examples/0*.py` script, so the UI and the standalone scripts share one definition.
Building happens at import time; the OpenAI key is only needed when an agent is
actually invoked.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

# .../examples (this file lives at examples/ui/backend/agents.py)
_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2]


def _load(module_name: str, filename: str) -> ModuleType:
    """Import an example script by file path under a clean module name.

    Args:
        module_name: Name to register the loaded module under.
        filename: Script file name inside the `examples/` directory.

    Returns:
        The imported module.
    """
    path = _EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"Could not load example module from {path}."
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_react_mod = _load("ex_react", "01_react_agent.py")
_struct_mod = _load("ex_struct", "02_structured_extractor.py")
_memory_mod = _load("ex_memory", "03_memory_chatbot.py")
_planner_mod = _load("ex_planner", "04_planner_agent.py")

react_agent = _react_mod.build_agent()
structured_agent = _struct_mod.build_agent()
memory_agent = _memory_mod.build_agent()
planner_agent = _planner_mod.build_agent()

# Pydantic schema produced by the structured extractor, re-exported for typing.
Person = _struct_mod.Person
