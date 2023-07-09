from typing import List, Tuple

from langchainplus_sdk.schemas import Run, RunTypeEnum

from langchain.schema.agent import AgentAction


def assemble_agent_trajectory(
    run: Run, *, tool_input_key: str = "input", tool_output_key: str = "output"
) -> List[Tuple[AgentAction, str]]:
    """Extract the series of steps from a run."""
    if run.child_runs is None:
        raise ValueError("Run must have child runs to be evaluated.")
    tool_runs = [run_ for run_ in run.child_runs if run_.run_type == RunTypeEnum.tool]
    agent_steps = []
    for run_ in tool_runs:
        tool_output = run_.outputs[tool_output_key] if run_.outputs else run_.error
        agent_steps.append(
            (
                AgentAction(
                    tool=run_.name, tool_input=run_.inputs[tool_input_key], log=""
                ),
                tool_output,
            )
        )
    return agent_steps
