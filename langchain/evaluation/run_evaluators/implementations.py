from typing import Any, Dict, Mapping, Optional, Sequence, Union

from langchainplus_sdk.evaluation import EvaluationResult
from langchainplus_sdk.schemas import Example, Run, RunTypeEnum
from pydantic import BaseModel, Field

from langchain.chat_models.base import BaseChatModel
from langchain.evaluation.agents.trajectory_eval_chain import (
    TrajectoryEvalChain,
    TrajectoryOutputParser,
)
from langchain.evaluation.criteria.eval_chain import (
    CriteriaEvalChain,
    CriteriaResultOutputParser,
)
from langchain.evaluation.qa.eval_chain import QAEvalChain
from langchain.evaluation.qa.eval_prompt import PROMPT as QA_DEFAULT_PROMPT
from langchain.evaluation.qa.eval_prompt import SQL_PROMPT
from langchain.evaluation.run_evaluators.base import (
    RunEvaluatorChain,
    RunEvaluatorInputMapper,
    RunEvaluatorOutputParser,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool

_QA_PROMPTS = {
    "qa": QA_DEFAULT_PROMPT,
    "sql": SQL_PROMPT,
}


class StringRunEvaluatorInputMapper(RunEvaluatorInputMapper, BaseModel):
    """Maps the Run and Optional[Example] to a dictionary."""

    prediction_map: Dict[str, str]
    """Map from run outputs to the evaluation inputs."""
    input_map: Dict[str, str]
    """Map from run inputs to the evaluation inputs."""
    answer_map: Optional[Dict[str, str]] = None
    """Map from example outputs to the evaluation inputs."""

    def map(self, run: Run, example: Optional[Example] = None) -> Dict[str, Any]:
        """Maps the Run and Optional[Example] to a dictionary"""
        if run.outputs is None and self.prediction_map:
            raise ValueError(f"Run {run.id} has no outputs.")
        if self.answer_map and (not example or not example.outputs):
            raise ValueError("This evaluator requires references, but none were given.")
        outputs = run.outputs or {}
        data = {value: outputs[key] for key, value in self.prediction_map.items()}
        data.update({value: run.inputs[key] for key, value in self.input_map.items()})
        if self.answer_map and example and example.outputs:
            data.update(
                {value: example.outputs[key] for key, value in self.answer_map.items()}
            )
        return data


class ChoicesOutputParser(RunEvaluatorOutputParser):
    """Parse a feedback run with optional choices."""

    evaluation_name: str
    choices_map: Optional[Dict[str, int]] = None

    @property
    def _type(self) -> str:
        return "choices_run_eval"

    def parse(self, text: str) -> EvaluationResult:
        """Parse the last line of the text and return an evaluation result."""
        lines = text.strip().split()
        value = lines[-1].strip()
        score = self.choices_map.get(value) if self.choices_map else None
        comment = " ".join(lines[:-1]) if len(lines) > 1 else None
        return EvaluationResult(
            key=self.evaluation_name,
            score=score,
            value=value,
            comment=comment,
        )


def get_qa_evaluator(
    llm: BaseLanguageModel,
    *,
    prompt: Union[PromptTemplate, str] = QA_DEFAULT_PROMPT,
    input_key: str = "input",
    prediction_key: str = "output",
    answer_key: str = "output",
    evaluation_name: Optional[str] = None,
    **kwargs: Any,
) -> RunEvaluatorChain:
    """Get an eval chain that compares response against ground truth."""
    if isinstance(prompt, str):
        prompt = _QA_PROMPTS[prompt]
    eval_chain = QAEvalChain.from_llm(llm=llm, prompt=prompt, **kwargs)
    input_mapper = kwargs.pop(
        "input_mapper",
        StringRunEvaluatorInputMapper(
            input_map={input_key: "query"},
            prediction_map={prediction_key: "result"},
            answer_map={answer_key: "answer"},
        ),
    )
    evaluation_name = evaluation_name or "Correctness"
    output_parser = kwargs.pop(
        "output_parser",
        ChoicesOutputParser(
            evaluation_name=evaluation_name,
            choices_map={"CORRECT": 1, "INCORRECT": 0},
        ),
    )
    tags = kwargs.pop("tags", [])
    return RunEvaluatorChain(
        eval_chain=eval_chain,
        input_mapper=input_mapper,
        output_parser=output_parser,
        tags=tags + [evaluation_name],
        **kwargs,
    )


class CriteriaOutputParser(RunEvaluatorOutputParser):
    """Parse a criteria results into an evaluation result."""

    evaluation_name: str

    @property
    def _type(self) -> str:
        return "criteria"

    def parse(self, parsed_output: Union[str, dict]) -> EvaluationResult:
        """Parse the last line of the text and return an evaluation result."""
        if isinstance(parsed_output, str):
            parsed_output_ = CriteriaResultOutputParser().parse(parsed_output)
        else:
            parsed_output_ = parsed_output
        return EvaluationResult(
            key=self.evaluation_name,
            score=parsed_output_["score"],
            value=parsed_output_["value"],
            comment=parsed_output_["reasoning"],
        )


def get_criteria_evaluator(
    llm: BaseLanguageModel,
    criteria: Union[Mapping[str, str], Sequence[str], str],
    *,
    input_key: str = "input",
    prediction_key: str = "output",
    prompt: Optional[BasePromptTemplate] = None,
    evaluation_name: Optional[str] = None,
    requires_reference: bool = False,
    **kwargs: Any,
) -> RunEvaluatorChain:
    """Get an eval chain for grading a model's response against a map of criteria."""
    input_mapper = kwargs.pop(
        "input_mapper",
        StringRunEvaluatorInputMapper(
            input_map={input_key: "input"},
            prediction_map={prediction_key: "output"},
        ),
    )
    criteria_ = CriteriaEvalChain.resolve_criteria(criteria)
    evaluation_name = evaluation_name or " ".join(criteria_.keys())
    parser = kwargs.pop(
        "output_parser",
        CriteriaOutputParser(
            choices_map={"Y": 1, "N": 0}, evaluation_name=evaluation_name
        ),
    )
    tags = kwargs.pop("tags", [])
    eval_chain = CriteriaEvalChain.from_llm(
        llm=llm,
        criteria=criteria_,
        prompt=prompt,
        requires_reference=requires_reference,
        **kwargs,
    )
    return RunEvaluatorChain(
        eval_chain=eval_chain,
        input_mapper=input_mapper,
        output_parser=parser,
        tags=tags + [evaluation_name],
        **kwargs,
    )


class TrajectoryRunEvalOutputParser(RunEvaluatorOutputParser, TrajectoryOutputParser):
    evaluation_name: str = "Agent Trajectory"
    """The name assigned to the evaluation feedback."""
    evaluator_info: dict = Field(default_factory=dict)
    """Additional information to log as feedback metadata."""

    @property
    def _type(self) -> str:
        return "agent_trajectory_run_eval"

    def parse_chain_output(self, output: Dict[str, Any]) -> EvaluationResult:
        """Parse the output of a run."""
        return EvaluationResult(
            key=self.evaluation_name,
            score=int(output["score"]),
            comment=output["reasoning"],
            evaluator_info=self.evaluator_info,
        )


class TrajectoryInputMapper(RunEvaluatorInputMapper, BaseModel):
    """Maps the Run and Optional[Example] to a dictionary."""

    agent_input_key: str = "input"
    """The key to load from the agent executor's run input dictionary."""
    agent_output_key: str = "output"
    """The key to load from the agent executor's run output dictionary."""
    tool_input_key: str = "input"
    """The key to load from the tool executor's run input dictionary."""
    tool_output_key: str = "output"
    """The key to load from the tool executor's run output dictionary."""
    reference_output_key: Optional[str] = None
    """The key to use for selecting the reference answer."""

    def map(self, run: Run, example: Optional[Example] = None) -> Dict[str, str]:
        """Maps the Run and Optional[Example] to a dictionary"""
        if run.child_runs is None:
            raise ValueError("Run must have child runs to be evaluated.")
        if run.outputs is None:
            raise ValueError("Run must have outputs to be evaluated.")
        reference = ""
        if example is not None and example.outputs:
            if self.reference_output_key is not None:
                reference = example.outputs[self.reference_output_key]
            elif "output" in example.outputs:
                reference = example.outputs["output"]
            elif len(example.outputs) == 1:
                reference = next(iter(example.outputs.values()))
            else:
                raise ValueError("Could not infer the reference answer from ")

        question = run.inputs[self.agent_input_key]
        tool_runs = [
            run_ for run_ in run.child_runs if run_.run_type == RunTypeEnum.tool
        ]
        agent_steps = []
        for i, run_ in enumerate(tool_runs, 1):
            tool_output = (
                f"Tool output: {run_.outputs.get(self.tool_output_key, run_.outputs)}"
                if run_.outputs
                else (f"Tool error: {run_.error}" if run_.error else "No output")
            )
            agent_steps.append(
                f"""Step {i}:
Tool used: {run_.name}
Tool input: {run_.inputs.get(self.tool_input_key, run_.inputs)}
Tool output: {tool_output}"""
            )

        return {
            "question": question,
            "agent_trajectory": "\n\n".join(agent_steps),
            "answer": run.outputs[self.agent_output_key],
            "reference": reference,
        }


def get_trajectory_evaluator(
    llm: BaseChatModel,
    agent_tools: Sequence[BaseTool],
    *,
    input_key: str = "input",
    prediction_key: str = "output",
    tool_input_key: str = "input",
    tool_output_key: str = "output",
    reference_output_key: Optional[str] = None,
    evaluation_name: str = "Agent Trajectory",
    **kwargs: Any,
) -> RunEvaluatorChain:
    """Get an eval chain for grading a model's response against a map of criteria."""
    input_mapper = kwargs.pop(
        "input_mapper",
        TrajectoryInputMapper(
            agent_input_key=input_key,
            agent_output_key=prediction_key,
            tool_input_key=tool_input_key,
            tool_output_key=tool_output_key,
            reference_output_key=reference_output_key,
        ),
    )
    parser = kwargs.pop(
        "output_parser",
        TrajectoryRunEvalOutputParser(evaluation_name=evaluation_name),
    )
    eval_chain = TrajectoryEvalChain.from_llm(
        llm=llm, agent_tools=agent_tools, return_reasoning=True, **kwargs
    )
    tags = kwargs.pop("tags", [])
    return RunEvaluatorChain(
        eval_chain=eval_chain,
        input_mapper=input_mapper,
        output_parser=parser,
        tags=tags + [evaluation_name],
        **kwargs,
    )
