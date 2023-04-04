import hashlib
import json
import random
import string
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from langchain.utils import get_from_dict_or_env


def import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "To use the mlflow callback manager you need to have the `mlflow` python "
            "package installed. Please install it with `pip install mlflow==2.2.2`"
        )
    return mlflow


def import_spacy() -> Any:
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "To use the mlflow callback manager you need to have the `spacy` python "
            "package installed. Please install it with `pip install spacy`"
        )
    return spacy


def import_pandas() -> Any:
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "To use the mlflow callback manager you need to have the `pandas` python "
            "package installed. Please install it with `pip install pandas`"
        )
    return pandas


def import_textstat() -> Any:
    try:
        import textstat
    except ImportError:
        raise ImportError(
            "To use the mlflow callback manager you need to have the `textstat` python "
            "package installed. Please install it with `pip install textstat`"
        )
    return textstat


def _flatten_dict(
    nested_dict: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Iterable[Tuple[str, Any]]:
    """
    Generator that yields flattened items from a nested dictionary for a flat dict.

    Parameters:
        nested_dict (dict): The nested dictionary to flatten.
        parent_key (str): The prefix to prepend to the keys of the flattened dict.
        sep (str): The separator to use between the parent key and the key of the
            flattened dictionary.

    Yields:
        (str, any): A key-value pair from the flattened dictionary.
    """
    for key, value in nested_dict.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            yield from _flatten_dict(value, new_key, sep)
        else:
            yield new_key, value


def flatten_dict(
    nested_dict: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    """Flattens a nested dictionary into a flat dictionary.

    Parameters:
        nested_dict (dict): The nested dictionary to flatten.
        parent_key (str): The prefix to prepend to the keys of the flattened dict.
        sep (str): The separator to use between the parent key and the key of the
            flattened dictionary.

    Returns:
        (dict): A flat dictionary.

    """
    flat_dict = {k: v for k, v in _flatten_dict(nested_dict, parent_key, sep)}
    return flat_dict


def hash_string(s: str) -> str:
    """Hash a string using sha1.

    Parameters:
        s (str): The string to hash.

    Returns:
        (str): The hashed string.
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def analyze_text(
    text: str,
    nlp: Any = None,
) -> dict:
    """Analyze text using textstat and spacy.

    Parameters:
        text (str): The text to analyze.
        nlp (spacy.lang): The spacy language model to use for visualization.

    Returns:
        (dict): A dictionary containing the complexity metrics and visualization
            files serialized to  HTML string.
    """
    resp = {}
    textstat = import_textstat()
    spacy = import_spacy()
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
    resp.update({"text_complexity_metrics": text_complexity_metrics})
    resp.update(text_complexity_metrics)

    if nlp is not None:
        doc = nlp(text)

        dep_out = spacy.displacy.render(  # type: ignore
            doc, style="dep", jupyter=False, page=True
        )

        ent_out = spacy.displacy.render(  # type: ignore
            doc, style="ent", jupyter=False, page=True
        )

        text_visualizations = {
            "dependency_tree": dep_out,
            "entities": ent_out,
        }

        resp.update(text_visualizations)

    return resp


def construct_html_from_prompt_and_generation(prompt: str, generation: str) -> Any:
    """Construct an html element from a prompt and a generation.

    Parameters:
        prompt (str): The prompt.
        generation (str): The generation.

    Returns:
        (str): The html string."""
    formatted_prompt = prompt.replace("\n", "<br>")
    formatted_generation = generation.replace("\n", "<br>")

    return f"""
    <p style="color:black;">{formatted_prompt}:</p>
    <blockquote>
      <p style="color:green;">
        {formatted_generation}
      </p>
    </blockquote>
    """


class MlflowLogger:
    """Callback Handler that logs metrics and artifacts to mlflow server.

    Parameters:
        name (str): Name of the run.
        experiment (str): Name of the experiment.
        tags (str): Tags to be attached for the run.
        tracking_uri (str): MLflow tracking server uri.

    This handler implements the helper functions to initialize,
    log metrics and artifacts to the mlflow server.
    """

    def __init__(self, **kwargs):
        mlflow = import_mlflow()
        tracking_uri = get_from_dict_or_env(
            kwargs, "tracking_uri", "MLFLOW_TRACKING_URI"
        )
        self.mlfc = mlflow.MlflowClient(tracking_uri=tracking_uri)

        # User can set other env variables described here
        # > https://www.mlflow.org/docs/latest/tracking.html#logging-to-a-tracking-server

        experiment_name = get_from_dict_or_env(
            kwargs, "experiment_name", "MLFLOW_EXPERIMENT_NAME"
        )
        self.mlf_exp = self.mlfc.get_experiment_by_name(experiment_name)
        if self.mlf_exp is not None:
            self.mlf_exp_id = self.mlf_exp.experiment_id
        else:
            self.mlf_exp_id = self.mlfc.create_experiment(experiment_name)
            self.mlf_exp = self.mlfc.get_experiment_by_name(experiment_name)

        self.start_run(kwargs["run_name"], kwargs["run_tags"])

    def start_run(self, name: str, tags: list = []):
        """To start a new run, auto generates the random suffix for name"""
        if name.endswith("-%"):
            rname = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))
            name = name.replace("%", rname)
        self.run = self.mlfc.create_run(self.mlf_exp_id, run_name=name)

    def finish_run(self):
        """To finish the run."""
        self.mlfc.set_terminated(self.run.info.run_id)

    def metric(self, key: str, value: float):
        """To log metric to mlflow server."""
        # metric value only of type float or int
        if not isinstance(value, float) and not isinstance(value, int):
            return
        self.mlfc.log_metric(self.run.info.run_id, key, value)

    def json(self, data: Any):
        """To log all metrics in the input dict."""
        for k, v in data.items():
            self.metric(k, v)

    def jsonf(self, data: Any, filename: str):
        """To log the input data as json file artifact."""
        self.mlfc.log_dict(self.run.info.run_id, data, f"{filename}.json")

    def table(self, name: str, df):
        """To log the input dataframe as a html table"""
        self.html(df.to_html(), f"table_{name}")

    def html(self, html: str, filename: str):
        """To log the input html string as html file artifact."""
        self.mlfc.log_text(self.run.info.run_id, html, f"{filename}.html")

    def text(self, text: str, filename: str):
        """To log the input text as text file artifact."""
        self.mlfc.log_text(self.run.info.run_id, text, f"{filename}.txt")

    def artifact(self, path: str):
        """To upload the file from given path as artifact."""
        self.mlfc.log_artifact(self.run.info.run_id, path)


class MlflowCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs metrics and artifacts to mlflow server.

    Parameters:
        name (str): Name of the run.
        experiment (str): Name of the experiment.
        tags (str): Tags to be attached for the run.
        tracking_uri (str): MLflow tracking server uri.

    This handler will utilize the associated callback method called and formats
    the input of each callback function with metadata regarding the state of LLM run,
    and adds the response to the list of records for both the {method}_records and
    action. It then logs the response to mlflow server.
    """

    def __init__(
        self,
        name: Optional[str] = "langchainrun-%",
        experiment: Optional[str] = "langchain",
        tags: Optional[Sequence] = [],
        tracking_uri: Optional[str] = None,
    ) -> None:
        """Initialize callback handler."""
        import_pandas()
        import_textstat()
        spacy = import_spacy()
        super().__init__()

        self.name = name
        self.experiment = experiment
        self.tags = tags
        self.tracking_uri = tracking_uri

        self.temp_dir = tempfile.TemporaryDirectory()

        self.mlflg = MlflowLogger(
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment,
            run_name=self.name,
            run_tags=self.tags,
        )

        self.action_records: list = []
        self.nlp = spacy.load("en_core_web_sm")

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

        self.records = {
            "on_llm_start_records": [],
            "on_llm_token_records": [],
            "on_llm_end_records": [],
            "on_chain_start_records": [],
            "on_chain_end_records": [],
            "on_tool_start_records": [],
            "on_tool_end_records": [],
            "on_text_records": [],
            "on_agent_finish_records": [],
            "on_agent_action_records": [],
            "action_records": [],
        }

    def _reset(self):
        for k, v in self.metrics.items():
            self.metrics[k] = 0
        for k, v in self.records.items():
            self.records[k] = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        self.metrics["step"] += 1
        self.metrics["llm_starts"] += 1
        self.metrics["starts"] += 1

        llm_starts = self.metrics["llm_starts"]

        resp = {}
        resp.update({"action": "on_llm_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        for idx, prompt in enumerate(prompts):
            prompt_resp = deepcopy(resp)
            prompt_resp["prompt"] = prompt
            self.records["on_llm_start_records"].append(prompt_resp)
            self.records["action_records"].append(prompt_resp)
            self.mlflg.jsonf(prompt_resp, f"llm_start_{llm_starts}_prompt_{idx}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        self.metrics["step"] += 1
        self.metrics["llm_streams"] += 1

        llm_streams = self.metrics["llm_streams"]

        resp = {}
        resp.update({"action": "on_llm_new_token", "token": token})
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        self.records["on_llm_token_records"].append(resp)
        self.records["action_records"].append(resp)
        self.mlflg.jsonf(resp, f"llm_new_tokens_{llm_streams}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.metrics["step"] += 1
        self.metrics["llm_ends"] += 1
        self.metrics["ends"] += 1

        llm_ends = self.metrics["llm_ends"]

        resp = {}
        resp.update({"action": "on_llm_end"})
        resp.update(flatten_dict(response.llm_output or {}))
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        for generations in response.generations:
            for idx, generation in enumerate(generations):
                generation_resp = deepcopy(resp)
                generation_resp.update(flatten_dict(generation.dict()))
                generation_resp.update(
                    analyze_text(
                        generation.text,
                        nlp=self.nlp,
                    )
                )
                self.mlflg.json(generation_resp.pop("text_complexity_metrics"))
                self.records["on_llm_end_records"].append(generation_resp)
                self.records["action_records"].append(generation_resp)
                self.mlflg.jsonf(resp, f"llm_end_{llm_ends}_generation_{idx}")
                dependency_tree = generation_resp["dependency_tree"]
                entities = generation_resp["entities"]
                self.mlflg.html(dependency_tree, "dep-" + hash_string(generation.text))
                self.mlflg.html(entities, "ent-" + hash_string(generation.text))

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
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

        resp = {}
        resp.update({"action": "on_chain_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        chain_input = inputs["input"]

        if isinstance(chain_input, str):
            input_resp = deepcopy(resp)
            input_resp["input"] = chain_input
            self.records["on_chain_start_records"].append(input_resp)
            self.records["action_records"].append(input_resp)
            self.mlflg.jsonf(input_resp, f"chain_start_{chain_starts}")
        elif isinstance(chain_input, list):
            for idx, inp in enumerate(chain_input):
                input_resp = deepcopy(resp)
                input_resp.update(inp)
                self.records["on_chain_start_records"].append(input_resp)
                self.records["action_records"].append(input_resp)
                self.mlflg.jsonf(input_resp, f"chain_start_{chain_starts}_input_{idx}")
        else:
            raise ValueError("Unexpected data format provided!")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.metrics["step"] += 1
        self.metrics["chain_ends"] += 1
        self.metrics["ends"] += 1

        chain_ends = self.metrics["chain_ends"]

        resp = {}
        resp.update({"action": "on_chain_end", "outputs": outputs["output"]})
        resp.update(self.metrics)

        self.mlfg.json(self.metrics)

        self.records["on_chain_end_records"].append(resp)
        self.records["action_records"].append(resp)
        self.mlfg.jsonf(resp, f"chain_end_{chain_ends}")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
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

        resp = {}
        resp.update({"action": "on_tool_start", "input_str": input_str})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        self.records["on_tool_start_records"].append(resp)
        self.records["action_records"].append(resp)
        self.mlflg.jsonf(resp, f"tool_start_{tool_starts}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.metrics["step"] += 1
        self.metrics["tool_ends"] += 1
        self.metrics["ends"] += 1

        tool_ends = self.metrics["tool_ends"]

        resp = {}
        resp.update({"action": "on_tool_end", "output": output})
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        self.records["on_tool_end_records"].append(resp)
        self.records["action_records"].append(resp)
        self.mlflg.jsonf(resp, f"tool_end_{tool_ends}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
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

        resp = {}
        resp.update({"action": "on_text", "text": text})
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        self.records["on_text_records"].append(resp)
        self.records["action_records"].append(resp)
        self.mlflg.jsonf(resp, f"on_text_{text_ctr}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        self.metrics["step"] += 1
        self.metrics["agent_ends"] += 1
        self.metrics["ends"] += 1

        agent_ends = self.metrics["agent_ends"]
        resp = {}
        resp.update(
            {
                "action": "on_agent_finish",
                "output": finish.return_values["output"],
                "log": finish.log,
            }
        )
        resp.update(self.metrics)

        self.mlflg.json(self.metrics)

        self.records["on_agent_finish_records"].append(resp)
        self.records["action_records"].append(resp)
        self.mlflg.jsonf(resp, f"agent_finish_{agent_ends}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.metrics["step"] += 1
        self.metrics["tool_starts"] += 1
        self.metrics["starts"] += 1

        tool_starts = self.metrics["tool_starts"]
        resp = {}
        resp.update(
            {
                "action": "on_agent_action",
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
        )
        resp.update(self.metrics)
        self.mlflg.json(metrics)
        self.records["on_agent_action_records"].append(resp)
        self.records["action_records"].append(resp)
        self.mlflg.jsonf(resp, f"agent_action_{tool_starts}")

    def _create_session_analysis_df(self) -> Any:
        """Create a dataframe with all the information from the session."""
        pd = import_pandas()
        on_llm_start_records_df = pd.DataFrame(self.records["on_llm_start_records"])
        on_llm_end_records_df = pd.DataFrame(self.records["on_llm_end_records"])

        llm_input_prompts_df = (
            on_llm_start_records_df[["step", "prompt", "name"]]
            .dropna(axis=1)
            .rename({"step": "prompt_step"}, axis=1)
        )
        complexity_metrics_columns = []
        visualizations_columns = []

        complexity_metrics_columns = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "smog_index",
            "coleman_liau_index",
            "automated_readability_index",
            "dale_chall_readability_score",
            "difficult_words",
            "linsear_write_formula",
            "gunning_fog",
            "text_standard",
            "fernandez_huerta",
            "szigriszt_pazos",
            "gutierrez_polini",
            "crawford",
            "gulpease_index",
            "osman",
        ]

        visualizations_columns = ["dependency_tree", "entities"]

        llm_outputs_df = (
            on_llm_end_records_df[
                [
                    "step",
                    "text",
                    "token_usage_total_tokens",
                    "token_usage_prompt_tokens",
                    "token_usage_completion_tokens",
                ]
                + complexity_metrics_columns
                + visualizations_columns
            ]
            .dropna(axis=1)
            .rename({"step": "output_step", "text": "output"}, axis=1)
        )
        session_analysis_df = pd.concat([llm_input_prompts_df, llm_outputs_df], axis=1)
        session_analysis_df["chat_html"] = session_analysis_df[
            ["prompt", "output"]
        ].apply(
            lambda row: construct_html_from_prompt_and_generation(
                row["prompt"], row["output"]
            ),
            axis=1,
        )
        return session_analysis_df

    def flush_tracker(self, langchain_asset: Any = None, finish: bool = False):

        pd = import_pandas()
        self.mlflg.table("action_records", pd.DataFrame(self.records["action_records"]))
        session_analysis_df = self._create_session_analysis_df()
        chat_html = session_analysis_df.pop("chat_html")
        chat_html = chat_html.replace("\n", "", regex=True)
        self.mlflg.table("session_analysis", pd.DataFrame(session_analysis_df))
        self.mlflg.html("".join(chat_html.tolist()), "chat_html")

        if langchain_asset:
            langchain_asset_path = Path(self.temp_dir.name, "model.json")
            try:
                langchain_asset.save(langchain_asset_path)
            except ValueError:
                langchain_asset.save_agent(langchain_asset_path)
            except NotImplementedError as e:
                print("Could not save model.")
                print(repr(e))
                pass
            self.mlflg.artifact(langchain_asset_path)
        if finish:
            self.mlflg.finish_run()
            self._reset()
