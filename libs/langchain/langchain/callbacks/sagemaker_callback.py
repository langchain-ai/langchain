import json
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.utils import (
    flatten_dict,
)
from langchain.schema import AgentAction, AgentFinish, LLMResult


def save_json(data: dict, file_path: str) -> None:
    """Save dict to local file path.

    Parameters:
        data (dict): The dictionary to be saved.
        file_path (str): Local file path.
    """
    with open(file_path, "w") as outfile:
        json.dump(data, outfile)


class SageMakerCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs prompt artifacts and metrics to SageMaker Experiments.

    Parameters:
        run (sagemaker.experiments.run.Run): Run object where the experiment is logged.
    """

    def __init__(self, run: Any) -> None:
        """Initialize callback handler."""
        super().__init__()

        self.run = run

        self.metrics = {
            "step": 0,
            "starts": 0,
            "ends": 0,
            "errors": 0,
            "text_ctr": 0,
            "chain_starts": 0,
            "chain_ends": 0,
            "llm_starts": 0,
            "llm_ends": 0,
            "llm_streams": 0,
            "tool_starts": 0,
            "tool_ends": 0,
            "agent_ends": 0,
        }

        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def _reset(self) -> None:
        for k, v in self.metrics.items():
            self.metrics[k] = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        self.metrics["step"] += 1
        self.metrics["llm_starts"] += 1
        self.metrics["starts"] += 1

        llm_starts = self.metrics["llm_starts"]

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_llm_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        for idx, prompt in enumerate(prompts):
            prompt_resp = deepcopy(resp)
            prompt_resp["prompt"] = prompt
            self.jsonf(
                prompt_resp,
                self.temp_dir,
                f"llm_start_{llm_starts}_prompt_{idx}",
            )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        self.metrics["step"] += 1
        self.metrics["llm_streams"] += 1

        llm_streams = self.metrics["llm_streams"]

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_llm_new_token", "token": token})
        resp.update(self.metrics)

        self.jsonf(resp, self.temp_dir, f"llm_new_tokens_{llm_streams}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.metrics["step"] += 1
        self.metrics["llm_ends"] += 1
        self.metrics["ends"] += 1

        llm_ends = self.metrics["llm_ends"]

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_llm_end"})
        resp.update(flatten_dict(response.llm_output or {}))

        resp.update(self.metrics)

        for generations in response.generations:
            for idx, generation in enumerate(generations):
                generation_resp = deepcopy(resp)
                generation_resp.update(flatten_dict(generation.dict()))

                self.jsonf(
                    resp,
                    self.temp_dir,
                    f"llm_end_{llm_ends}_generation_{idx}",
                )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.metrics["step"] += 1
        self.metrics["errors"] += 1

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        self.metrics["step"] += 1
        self.metrics["chain_starts"] += 1
        self.metrics["starts"] += 1

        chain_starts = self.metrics["chain_starts"]

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_chain_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        chain_input = ",".join([f"{k}={v}" for k, v in inputs.items()])
        input_resp = deepcopy(resp)
        input_resp["inputs"] = chain_input

        self.jsonf(input_resp, self.temp_dir, f"chain_start_{chain_starts}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.metrics["step"] += 1
        self.metrics["chain_ends"] += 1
        self.metrics["ends"] += 1

        chain_ends = self.metrics["chain_ends"]

        resp: Dict[str, Any] = {}
        chain_output = ",".join([f"{k}={v}" for k, v in outputs.items()])
        resp.update({"action": "on_chain_end", "outputs": chain_output})
        resp.update(self.metrics)

        self.jsonf(resp, self.temp_dir, f"chain_end_{chain_ends}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors."""
        self.metrics["step"] += 1
        self.metrics["errors"] += 1

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        self.metrics["step"] += 1
        self.metrics["tool_starts"] += 1
        self.metrics["starts"] += 1

        tool_starts = self.metrics["tool_starts"]

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_tool_start", "input_str": input_str})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        self.jsonf(resp, self.temp_dir, f"tool_start_{tool_starts}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.metrics["step"] += 1
        self.metrics["tool_ends"] += 1
        self.metrics["ends"] += 1

        tool_ends = self.metrics["tool_ends"]

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_tool_end", "output": output})
        resp.update(self.metrics)

        self.jsonf(resp, self.temp_dir, f"tool_end_{tool_ends}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors."""
        self.metrics["step"] += 1
        self.metrics["errors"] += 1

    def on_text(self, text: str, **kwargs: Any) -> None:
        """
        Run when agent is ending.
        """
        self.metrics["step"] += 1
        self.metrics["text_ctr"] += 1

        text_ctr = self.metrics["text_ctr"]

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_text", "text": text})
        resp.update(self.metrics)

        self.jsonf(resp, self.temp_dir, f"on_text_{text_ctr}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        self.metrics["step"] += 1
        self.metrics["agent_ends"] += 1
        self.metrics["ends"] += 1

        agent_ends = self.metrics["agent_ends"]
        resp: Dict[str, Any] = {}
        resp.update(
            {
                "action": "on_agent_finish",
                "output": finish.return_values["output"],
                "log": finish.log,
            }
        )
        resp.update(self.metrics)

        self.jsonf(resp, self.temp_dir, f"agent_finish_{agent_ends}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.metrics["step"] += 1
        self.metrics["tool_starts"] += 1
        self.metrics["starts"] += 1

        tool_starts = self.metrics["tool_starts"]
        resp: Dict[str, Any] = {}
        resp.update(
            {
                "action": "on_agent_action",
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
        )
        resp.update(self.metrics)
        self.jsonf(resp, self.temp_dir, f"agent_action_{tool_starts}")

    def jsonf(
        self,
        data: Dict[str, Any],
        data_dir: str,
        filename: str,
        is_output: Optional[bool] = True,
    ) -> None:
        """To log the input data as json file artifact."""
        file_path = os.path.join(data_dir, f"{filename}.json")
        save_json(data, file_path)
        self.run.log_file(file_path, name=filename, is_output=is_output)

    def flush_tracker(self) -> None:
        """Reset the steps and delete the temporary local directory."""
        self._reset()
        shutil.rmtree(self.temp_dir)
