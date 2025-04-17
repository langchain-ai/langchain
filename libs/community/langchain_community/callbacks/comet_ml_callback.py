import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
from langchain_core.utils import guard_import

import langchain_community
from langchain_community.callbacks.utils import (
    BaseMetadataCallbackHandler,
    flatten_dict,
    import_pandas,
    import_spacy,
    import_textstat,
)

LANGCHAIN_MODEL_NAME = "langchain-model"


def import_comet_ml() -> Any:
    """Import comet_ml and raise an error if it is not installed."""
    return guard_import("comet_ml")


def _get_experiment(
    workspace: Optional[str] = None, project_name: Optional[str] = None
) -> Any:
    comet_ml = import_comet_ml()

    experiment = comet_ml.Experiment(
        workspace=workspace,
        project_name=project_name,
    )

    return experiment


def _fetch_text_complexity_metrics(text: str) -> dict:
    textstat = import_textstat()
    text_complexity_metrics = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "difficult_words": textstat.difficult_words(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "gunning_fog": textstat.gunning_fog(text),
        "text_standard": textstat.text_standard(text),
        "fernandez_huerta": textstat.fernandez_huerta(text),
        "szigriszt_pazos": textstat.szigriszt_pazos(text),
        "gutierrez_polini": textstat.gutierrez_polini(text),
        "crawford": textstat.crawford(text),
        "gulpease_index": textstat.gulpease_index(text),
        "osman": textstat.osman(text),
    }
    return text_complexity_metrics


def _summarize_metrics_for_generated_outputs(metrics: Sequence) -> dict:
    pd = import_pandas()
    metrics_df = pd.DataFrame(metrics)
    metrics_summary = metrics_df.describe()

    return metrics_summary.to_dict()


class CometCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler):
    """Callback Handler that logs to Comet.

    Parameters:
        job_type (str): The type of comet_ml task such as "inference",
            "testing" or "qc"
        project_name (str): The comet_ml project name
        tags (list): Tags to add to the task
        task_name (str): Name of the comet_ml task
        visualize (bool): Whether to visualize the run.
        complexity_metrics (bool): Whether to log complexity metrics
        stream_logs (bool): Whether to stream callback actions to Comet

    This handler will utilize the associated callback method and formats
    the input of each callback function with metadata regarding the state of LLM run,
    and adds the response to the list of records for both the {method}_records and
    action. It then logs the response to Comet.
    """

    def __init__(
        self,
        task_type: Optional[str] = "inference",
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        visualizations: Optional[List[str]] = None,
        complexity_metrics: bool = False,
        custom_metrics: Optional[Callable] = None,
        stream_logs: bool = True,
    ) -> None:
        """Initialize callback handler."""

        self.comet_ml = import_comet_ml()
        super().__init__()

        self.task_type = task_type
        self.workspace = workspace
        self.project_name = project_name
        self.tags = tags
        self.visualizations = visualizations
        self.complexity_metrics = complexity_metrics
        self.custom_metrics = custom_metrics
        self.stream_logs = stream_logs
        self.temp_dir = tempfile.TemporaryDirectory()

        self.experiment = _get_experiment(workspace, project_name)
        self.experiment.log_other("Created from", "langchain")
        if tags:
            self.experiment.add_tags(tags)
        self.name = name
        if self.name:
            self.experiment.set_name(self.name)

        warning = (
            "The comet_ml callback is currently in beta and is subject to change "
            "based on updates to `langchain`. Please report any issues to "
            "https://github.com/comet-ml/issue-tracking/issues with the tag "
            "`langchain`."
        )
        self.comet_ml.LOGGER.warning(warning)

        self.callback_columns: list = []
        self.action_records: list = []
        self.complexity_metrics = complexity_metrics
        if self.visualizations:
            spacy = import_spacy()
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None

    def _init_resp(self) -> Dict:
        return {k: None for k in self.callback_columns}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        self.step += 1
        self.llm_starts += 1
        self.starts += 1

        metadata = self._init_resp()
        metadata.update({"action": "on_llm_start"})
        metadata.update(flatten_dict(serialized))
        metadata.update(self.get_custom_callback_meta())

        for prompt in prompts:
            prompt_resp = deepcopy(metadata)
            prompt_resp["prompts"] = prompt
            self.on_llm_start_records.append(prompt_resp)
            self.action_records.append(prompt_resp)

            if self.stream_logs:
                self._log_stream(prompt, metadata, self.step)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        self.step += 1
        self.llm_streams += 1

        resp = self._init_resp()
        resp.update({"action": "on_llm_new_token", "token": token})
        resp.update(self.get_custom_callback_meta())

        self.action_records.append(resp)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.step += 1
        self.llm_ends += 1
        self.ends += 1

        metadata = self._init_resp()
        metadata.update({"action": "on_llm_end"})
        metadata.update(flatten_dict(response.llm_output or {}))
        metadata.update(self.get_custom_callback_meta())

        output_complexity_metrics = []
        output_custom_metrics = []

        for prompt_idx, generations in enumerate(response.generations):
            for gen_idx, generation in enumerate(generations):
                text = generation.text

                generation_resp = deepcopy(metadata)
                generation_resp.update(flatten_dict(generation.dict()))

                complexity_metrics = self._get_complexity_metrics(text)
                if complexity_metrics:
                    output_complexity_metrics.append(complexity_metrics)
                    generation_resp.update(complexity_metrics)

                custom_metrics = self._get_custom_metrics(
                    generation, prompt_idx, gen_idx
                )
                if custom_metrics:
                    output_custom_metrics.append(custom_metrics)
                    generation_resp.update(custom_metrics)

                if self.stream_logs:
                    self._log_stream(text, metadata, self.step)

                self.action_records.append(generation_resp)
                self.on_llm_end_records.append(generation_resp)

        self._log_text_metrics(output_complexity_metrics, step=self.step)
        self._log_text_metrics(output_custom_metrics, step=self.step)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.step += 1
        self.errors += 1

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        self.step += 1
        self.chain_starts += 1
        self.starts += 1

        resp = self._init_resp()
        resp.update({"action": "on_chain_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())

        for chain_input_key, chain_input_val in inputs.items():
            if isinstance(chain_input_val, str):
                input_resp = deepcopy(resp)
                if self.stream_logs:
                    self._log_stream(chain_input_val, resp, self.step)
                input_resp.update({chain_input_key: chain_input_val})
                self.action_records.append(input_resp)

            else:
                self.comet_ml.LOGGER.warning(
                    f"Unexpected data format provided! "
                    f"Input Value for {chain_input_key} will not be logged"
                )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.step += 1
        self.chain_ends += 1
        self.ends += 1

        resp = self._init_resp()
        resp.update({"action": "on_chain_end"})
        resp.update(self.get_custom_callback_meta())

        for chain_output_key, chain_output_val in outputs.items():
            if isinstance(chain_output_val, str):
                output_resp = deepcopy(resp)
                if self.stream_logs:
                    self._log_stream(chain_output_val, resp, self.step)
                output_resp.update({chain_output_key: chain_output_val})
                self.action_records.append(output_resp)
            else:
                self.comet_ml.LOGGER.warning(
                    f"Unexpected data format provided! "
                    f"Output Value for {chain_output_key} will not be logged"
                )

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors."""
        self.step += 1
        self.errors += 1

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        self.step += 1
        self.tool_starts += 1
        self.starts += 1

        resp = self._init_resp()
        resp.update({"action": "on_tool_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())
        if self.stream_logs:
            self._log_stream(input_str, resp, self.step)

        resp.update({"input_str": input_str})
        self.action_records.append(resp)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running."""
        output = str(output)
        self.step += 1
        self.tool_ends += 1
        self.ends += 1

        resp = self._init_resp()
        resp.update({"action": "on_tool_end"})
        resp.update(self.get_custom_callback_meta())
        if self.stream_logs:
            self._log_stream(output, resp, self.step)

        resp.update({"output": output})
        self.action_records.append(resp)

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

        resp = self._init_resp()
        resp.update({"action": "on_text"})
        resp.update(self.get_custom_callback_meta())
        if self.stream_logs:
            self._log_stream(text, resp, self.step)

        resp.update({"text": text})
        self.action_records.append(resp)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        self.step += 1
        self.agent_ends += 1
        self.ends += 1

        resp = self._init_resp()
        output = finish.return_values["output"]
        log = finish.log

        resp.update({"action": "on_agent_finish", "log": log})
        resp.update(self.get_custom_callback_meta())
        if self.stream_logs:
            self._log_stream(output, resp, self.step)

        resp.update({"output": output})
        self.action_records.append(resp)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.step += 1
        self.tool_starts += 1
        self.starts += 1

        tool = action.tool
        tool_input = str(action.tool_input)
        log = action.log

        resp = self._init_resp()
        resp.update({"action": "on_agent_action", "log": log, "tool": tool})
        resp.update(self.get_custom_callback_meta())
        if self.stream_logs:
            self._log_stream(tool_input, resp, self.step)

        resp.update({"tool_input": tool_input})
        self.action_records.append(resp)

    def _get_complexity_metrics(self, text: str) -> dict:
        """Compute text complexity metrics using textstat.

        Parameters:
            text (str): The text to analyze.

        Returns:
            (dict): A dictionary containing the complexity metrics.
        """
        resp = {}
        if self.complexity_metrics:
            text_complexity_metrics = _fetch_text_complexity_metrics(text)
            resp.update(text_complexity_metrics)

        return resp

    def _get_custom_metrics(
        self, generation: Generation, prompt_idx: int, gen_idx: int
    ) -> dict:
        """Compute Custom Metrics for an LLM Generated Output

        Args:
            generation (LLMResult): Output generation from an LLM
            prompt_idx (int): List index of the input prompt
            gen_idx (int): List index of the generated output

        Returns:
            dict: A dictionary containing the custom metrics.
        """

        resp = {}
        if self.custom_metrics:
            custom_metrics = self.custom_metrics(generation, prompt_idx, gen_idx)
            resp.update(custom_metrics)

        return resp

    def flush_tracker(
        self,
        langchain_asset: Any = None,
        task_type: Optional[str] = "inference",
        workspace: Optional[str] = None,
        project_name: Optional[str] = "comet-langchain-demo",
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        visualizations: Optional[List[str]] = None,
        complexity_metrics: bool = False,
        custom_metrics: Optional[Callable] = None,
        finish: bool = False,
        reset: bool = False,
    ) -> None:
        """Flush the tracker and setup the session.

        Everything after this will be a new table.

        Args:
            name: Name of the performed session so far so it is identifiable
            langchain_asset: The langchain asset to save.
            finish: Whether to finish the run.

            Returns:
                None
        """
        self._log_session(langchain_asset)

        if langchain_asset:
            try:
                self._log_model(langchain_asset)
            except Exception:
                self.comet_ml.LOGGER.error(
                    "Failed to export agent or LLM to Comet",
                    exc_info=True,
                    extra={"show_traceback": True},
                )

        if finish:
            self.experiment.end()

        if reset:
            self._reset(
                task_type,
                workspace,
                project_name,
                tags,
                name,
                visualizations,
                complexity_metrics,
                custom_metrics,
            )

    def _log_stream(self, prompt: str, metadata: dict, step: int) -> None:
        self.experiment.log_text(prompt, metadata=metadata, step=step)

    def _log_model(self, langchain_asset: Any) -> None:
        model_parameters = self._get_llm_parameters(langchain_asset)
        self.experiment.log_parameters(model_parameters, prefix="model")

        langchain_asset_path = Path(self.temp_dir.name, "model.json")
        model_name = self.name if self.name else LANGCHAIN_MODEL_NAME

        try:
            if hasattr(langchain_asset, "save"):
                langchain_asset.save(langchain_asset_path)
                self.experiment.log_model(model_name, str(langchain_asset_path))
        except (ValueError, AttributeError, NotImplementedError) as e:
            if hasattr(langchain_asset, "save_agent"):
                langchain_asset.save_agent(langchain_asset_path)
                self.experiment.log_model(model_name, str(langchain_asset_path))
            else:
                self.comet_ml.LOGGER.error(
                    f"{e}"
                    " Could not save Langchain Asset "
                    f"for {langchain_asset.__class__.__name__}"
                )

    def _log_session(self, langchain_asset: Optional[Any] = None) -> None:
        try:
            llm_session_df = self._create_session_analysis_dataframe(langchain_asset)
            # Log the cleaned dataframe as a table
            self.experiment.log_table("langchain-llm-session.csv", llm_session_df)
        except Exception:
            self.comet_ml.LOGGER.warning(
                "Failed to log session data to Comet",
                exc_info=True,
                extra={"show_traceback": True},
            )

        try:
            metadata = {"langchain_version": str(langchain_community.__version__)}
            # Log the langchain low-level records as a JSON file directly
            self.experiment.log_asset_data(
                self.action_records, "langchain-action_records.json", metadata=metadata
            )
        except Exception:
            self.comet_ml.LOGGER.warning(
                "Failed to log session data to Comet",
                exc_info=True,
                extra={"show_traceback": True},
            )

        try:
            self._log_visualizations(llm_session_df)
        except Exception:
            self.comet_ml.LOGGER.warning(
                "Failed to log visualizations to Comet",
                exc_info=True,
                extra={"show_traceback": True},
            )

    def _log_text_metrics(self, metrics: Sequence[dict], step: int) -> None:
        if not metrics:
            return

        metrics_summary = _summarize_metrics_for_generated_outputs(metrics)
        for key, value in metrics_summary.items():
            self.experiment.log_metrics(value, prefix=key, step=step)

    def _log_visualizations(self, session_df: Any) -> None:
        if not (self.visualizations and self.nlp):
            return

        spacy = import_spacy()

        prompts = session_df["prompts"].tolist()
        outputs = session_df["text"].tolist()

        for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
            doc = self.nlp(output)
            sentence_spans = list(doc.sents)

            for visualization in self.visualizations:
                try:
                    html = spacy.displacy.render(
                        sentence_spans,
                        style=visualization,
                        options={"compact": True},
                        jupyter=False,
                        page=True,
                    )
                    self.experiment.log_asset_data(
                        html,
                        name=f"langchain-viz-{visualization}-{idx}.html",
                        metadata={"prompt": prompt},
                        step=idx,
                    )
                except Exception as e:
                    self.comet_ml.LOGGER.warning(
                        e, exc_info=True, extra={"show_traceback": True}
                    )

        return

    def _reset(
        self,
        task_type: Optional[str] = None,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        visualizations: Optional[List[str]] = None,
        complexity_metrics: bool = False,
        custom_metrics: Optional[Callable] = None,
    ) -> None:
        _task_type = task_type if task_type else self.task_type
        _workspace = workspace if workspace else self.workspace
        _project_name = project_name if project_name else self.project_name
        _tags = tags if tags else self.tags
        _name = name if name else self.name
        _visualizations = visualizations if visualizations else self.visualizations
        _complexity_metrics = (
            complexity_metrics if complexity_metrics else self.complexity_metrics
        )
        _custom_metrics = custom_metrics if custom_metrics else self.custom_metrics

        self.__init__(  # type: ignore[misc]
            task_type=_task_type,
            workspace=_workspace,
            project_name=_project_name,
            tags=_tags,
            name=_name,
            visualizations=_visualizations,
            complexity_metrics=_complexity_metrics,
            custom_metrics=_custom_metrics,
        )

        self.reset_callback_meta()
        self.temp_dir = tempfile.TemporaryDirectory()

    def _create_session_analysis_dataframe(self, langchain_asset: Any = None) -> dict:
        pd = import_pandas()

        llm_parameters = self._get_llm_parameters(langchain_asset)
        num_generations_per_prompt = llm_parameters.get("n", 1)

        llm_start_records_df = pd.DataFrame(self.on_llm_start_records)
        # Repeat each input row based on the number of outputs generated per prompt
        llm_start_records_df = llm_start_records_df.loc[
            llm_start_records_df.index.repeat(num_generations_per_prompt)
        ].reset_index(drop=True)
        llm_end_records_df = pd.DataFrame(self.on_llm_end_records)

        llm_session_df = pd.merge(
            llm_start_records_df,
            llm_end_records_df,
            left_index=True,
            right_index=True,
            suffixes=["_llm_start", "_llm_end"],
        )

        return llm_session_df

    def _get_llm_parameters(self, langchain_asset: Any = None) -> dict:
        if not langchain_asset:
            return {}
        try:
            if hasattr(langchain_asset, "agent"):
                llm_parameters = langchain_asset.agent.llm_chain.llm.dict()
            elif hasattr(langchain_asset, "llm_chain"):
                llm_parameters = langchain_asset.llm_chain.llm.dict()
            elif hasattr(langchain_asset, "llm"):
                llm_parameters = langchain_asset.llm.dict()
            else:
                llm_parameters = langchain_asset.dict()
        except Exception:
            return {}

        return llm_parameters
