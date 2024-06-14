"""**Autonomous agents** in the Langchain experimental package include
[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT),
[BabyAGI](https://github.com/yoheinakajima/babyagi),
and [HuggingGPT](https://arxiv.org/abs/2303.17580) agents that
interact with language models autonomously.

These agents have specific functionalities like memory management,
task creation, execution chains, and response generation.

They differ from ordinary agents by their autonomous decision-making capabilities,
memory handling, and specialized functionalities for tasks and response.
"""
from langchain_experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain_experimental.autonomous_agents.baby_agi.baby_agi import BabyAGI
from langchain_experimental.autonomous_agents.hugginggpt.hugginggpt import HuggingGPT

__all__ = ["BabyAGI", "AutoGPT", "HuggingGPT"]
