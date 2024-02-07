from copy import deepcopy
from typing import Any, Dict, List, Optional

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def import_aim() -> Any:
    """Import the aim python package and raise an error if it is not installed."""
    try:
        import aim
    except ImportError:
        raise ImportError(
            "To use the Aim callback manager you need to have the"
            " `aim` python package installed."
            "Please install it with `pip install aim`"
        )
    return aim


class BaseMetadataCallbackHandler:
    """This class handles the metadata and associated function states for callbacks.

    Attributes:
        step (int): The current step.
        starts (int): The number of times the start method has been called.
        ends (int): The number of times the end method has been called.
        errors (int): The number of times the error method has been called.
        text_ctr (int): The number of times the text method has been called.
        ignore_llm_ (bool): Whether to ignore llm callbacks.
        ignore_chain_ (bool): Whether to ignore chain callbacks.
        ignore_agent_ (bool): Whether to ignore agent callbacks.
        ignore_retriever_ (bool): Whether to ignore retriever callbacks.
        always_verbose_ (bool): Whether to always be verbose.
        chain_starts (int): The number of times the chain start method has been called.
        chain_ends (int): The number of times the chain end method has been called.
        llm_starts (int): The number of times the llm start method has been called.
        llm_ends (int): The number of times the llm end method has been called.
        llm_streams (int): The number of times the text method has been called.
        tool_starts (int): The number of times the tool start method has been called.
        tool_ends (int): The number of times the tool end method has been called.
        agent_ends (int): The number of times the agent end method has been called.
    """

    def __init__(self) -> None:
        self.step = 0

        self.starts = 0
        self.ends = 0
        self.errors = 0
        self.text_ctr = 0

        self.ignore_llm_ = False
        self.ignore_chain_ = False
        self.ignore_agent_ = False
        self.ignore_retriever_ = False
        self.always_verbose_ = False

        self.chain_starts = 0
        self.chain_ends = 0

        self.llm_starts = 0
        self.llm_ends = 0
        self.llm_streams = 0

        self.tool_starts = 0
        self.tool_ends = 0

        self.agent_ends = 0

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return self.always_verbose_

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return self.ignore_llm_

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return self.ignore_chain_

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return self.ignore_agent_

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return self.ignore_retriever_

    def get_custom_callback_meta(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "starts": self.starts,
            "ends": self.ends,
            "errors": self.errors,
            "text_ctr": self.text_ctr,
            "chain_starts": self.chain_starts,
            "chain_ends": self.chain_ends,
            "llm_starts": self.llm_starts,
            "llm_ends": self.llm_ends,
            "llm_streams": self.llm_streams,
            "tool_starts": self.tool_starts,
            "tool_ends": self.tool_ends,
            "agent_ends": self.agent_ends,
        }

    def reset_callback_meta(self) -> None:
        """Reset the callback metadata."""
        self.step = 0

        self.starts = 0
        self.ends = 0
        self.errors = 0
        self.text_ctr = 0

        self.ignore_llm_ = False
        self.ignore_chain_ = False
        self.ignore_agent_ = False
        self.always_verbose_ = False

        self.chain_starts = 0
        self.chain_ends = 0

        self.llm_starts = 0
        self.llm_ends = 0
        self.llm_streams = 0

        self.tool_starts = 0
        self.tool_ends = 0

        self.agent_ends = 0

        return None


class AimCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler):
    """Callback Handler that logs to Aim.

    Parameters:
        repo (:obj:`str`, optional): Aim repository path or Repo object to which
            Run object is bound. If skipped, default Repo is used.
        experiment_name (:obj:`str`, optional): Sets Run's `experiment` property.
            'default' if not specified. Can be used later to query runs/sequences.
        system_tracking_interval (:obj:`int`, optional): Sets the tracking interval
            in seconds for system usage metrics (CPU, Memory, etc.). Set to `None`
             to disable system metrics tracking.
        log_system_params (:obj:`bool`, optional): Enable/Disable logging of system
            params such as installed packages, git info, environment variables, etc.

    This handler will utilize the associated callback method called and formats
    the input of each callback function with metadata regarding the state of LLM run
    and then logs the response to Aim.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment_name: Optional[str] = None,
        system_tracking_interval: Optional[int] = 10,
        log_system_params: bool = True,
    ) -> None:
        """Initialize callback handler."""

        super().__init__()

        aim = import_aim()
        self.repo = repo
        self.experiment_name = experiment_name
        self.system_tracking_interval = system_tracking_interval
        self.log_system_params = log_system_params
        self._run = aim.Run(
            repo=self.repo,
            experiment=self.experiment_name,
            system_tracking_interval=self.system_tracking_interval,
            log_system_params=self.log_system_params,
        )
        self._run_hash = self._run.hash
        self.action_records: list = []

    def setup(self, **kwargs: Any) -> None:
        aim = import_aim()

        if not self._run:
            if self._run_hash:
                self._run = aim.Run(
                    self._run_hash,
                    repo=self.repo,
                    system_tracking_interval=self.system_tracking_interval,
                )
            else:
                self._run = aim.Run(
                    repo=self.repo,
                    experiment=self.experiment_name,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                )
                self._run_hash = self._run.hash

        if kwargs:
            for key, value in kwargs.items():
                self._run.set(key, value, strict=False)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        aim = import_aim()

        self.step += 1
        self.llm_starts += 1
        self.starts += 1

        resp = {"action": "on_llm_start"}
        resp.update(self.get_custom_callback_meta())

        prompts_res = deepcopy(prompts)

        self._run.track(
            [aim.Text(prompt) for prompt in prompts_res],
            name="on_llm_start",
            context=resp,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        aim = import_aim()
        self.step += 1
        self.llm_ends += 1
        self.ends += 1

        resp = {"action": "on_llm_end"}
        resp.update(self.get_custom_callback_meta())

        response_res = deepcopy(response)

        generated = [
            aim.Text(generation.text)
            for generations in response_res.generations
            for generation in generations
        ]
        self._run.track(
            generated,
            name="on_llm_end",
            context=resp,
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        self.step += 1
        self.llm_streams += 1

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.step += 1
        self.errors += 1

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        aim = import_aim()
        self.step += 1
        self.chain_starts += 1
        self.starts += 1

        resp = {"action": "on_chain_start"}
        resp.update(self.get_custom_callback_meta())

        inputs_res = deepcopy(inputs)

        self._run.track(
            aim.Text(inputs_res["input"]), name="on_chain_start", context=resp
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        aim = import_aim()
        self.step += 1
        self.chain_ends += 1
        self.ends += 1

        resp = {"action": "on_chain_end"}
        resp.update(self.get_custom_callback_meta())

        outputs_res = deepcopy(outputs)

        self._run.track(
            aim.Text(outputs_res["output"]), name="on_chain_end", context=resp
        )

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors."""
        self.step += 1
        self.errors += 1

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        aim = import_aim()
        self.step += 1
        self.tool_starts += 1
        self.starts += 1

        resp = {"action": "on_tool_start"}
        resp.update(self.get_custom_callback_meta())

        self._run.track(aim.Text(input_str), name="on_tool_start", context=resp)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        aim = import_aim()
        self.step += 1
        self.tool_ends += 1
        self.ends += 1

        resp = {"action": "on_tool_end"}
        resp.update(self.get_custom_callback_meta())

        self._run.track(aim.Text(output), name="on_tool_end", context=resp)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors."""
        self.step += 1
        self.errors += 1

    def on_text(self, text: str, **kwargs: Any) -> None:
        """
        Run when agent is ending.
        """
        self.step += 1
        self.text_ctr += 1

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        aim = import_aim()
        self.step += 1
        self.agent_ends += 1
        self.ends += 1

        resp = {"action": "on_agent_finish"}
        resp.update(self.get_custom_callback_meta())

        finish_res = deepcopy(finish)

        text = "OUTPUT:\n{}\n\nLOG:\n{}".format(
            finish_res.return_values["output"], finish_res.log
        )
        self._run.track(aim.Text(text), name="on_agent_finish", context=resp)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        aim = import_aim()
        self.step += 1
        self.tool_starts += 1
        self.starts += 1

        resp = {
            "action": "on_agent_action",
            "tool": action.tool,
        }
        resp.update(self.get_custom_callback_meta())

        action_res = deepcopy(action)

        text = "TOOL INPUT:\n{}\n\nLOG:\n{}".format(
            action_res.tool_input, action_res.log
        )
        self._run.track(aim.Text(text), name="on_agent_action", context=resp)

    def flush_tracker(
        self,
        repo: Optional[str] = None,
        experiment_name: Optional[str] = None,
        system_tracking_interval: Optional[int] = 10,
        log_system_params: bool = True,
        langchain_asset: Any = None,
        reset: bool = True,
        finish: bool = False,
    ) -> None:
        """Flush the tracker and reset the session.

        Args:
            repo (:obj:`str`, optional): Aim repository path or Repo object to which
                Run object is bound. If skipped, default Repo is used.
            experiment_name (:obj:`str`, optional): Sets Run's `experiment` property.
                'default' if not specified. Can be used later to query runs/sequences.
            system_tracking_interval (:obj:`int`, optional): Sets the tracking interval
                in seconds for system usage metrics (CPU, Memory, etc.). Set to `None`
                 to disable system metrics tracking.
            log_system_params (:obj:`bool`, optional): Enable/Disable logging of system
                params such as installed packages, git info, environment variables, etc.
            langchain_asset: The langchain asset to save.
            reset: Whether to reset the session.
            finish: Whether to finish the run.

            Returns:
                None
        """

        if langchain_asset:
            try:
                for key, value in langchain_asset.dict().items():
                    self._run.set(key, value, strict=False)
            except Exception:
                pass

        if finish or reset:
            self._run.close()
            self.reset_callback_meta()
        if reset:
            aim = import_aim()
            self.repo = repo if repo else self.repo
            self.experiment_name = (
                experiment_name if experiment_name else self.experiment_name
            )
            self.system_tracking_interval = (
                system_tracking_interval
                if system_tracking_interval
                else self.system_tracking_interval
            )
            self.log_system_params = (
                log_system_params if log_system_params else self.log_system_params
            )

            self._run = aim.Run(
                repo=self.repo,
                experiment=self.experiment_name,
                system_tracking_interval=self.system_tracking_interval,
                log_system_params=self.log_system_params,
            )
            self._run_hash = self._run.hash
            self.action_records = []
