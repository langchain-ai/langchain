# langchain-torchagentic

[LangChain](https://github.com/langchain-ai/langchain) integration for [torchagentic](https://github.com/liodon-ai/torchagentic), providing differentiable planning primitives (value iteration, MCTS) as LangChain tools.

## Installation

```bash
pip install -U langchain-torchagentic
```

## Usage

```python
from langchain_torchagentic import TorchAgenticPlannerTool

tool = TorchAgenticPlannerTool(
    num_states=64,
    num_actions=8,
    planner_type="vi",  # "vi" for value iteration, "mcts" for MCTS
)

result = tool.invoke({"task": "Find optimal route from A to B"})
print(result)
# [TorchAgentic | VI | top states: [3, 12, 13] | states: 64 | actions: 8 | task: Find optimal route from A to B]
```
