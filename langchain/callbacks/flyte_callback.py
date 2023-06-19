from copy import deepcopy
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.utils import (
    BaseMetadataCallbackHandler,
    flatten_dict,
    import_spacy,
    import_pandas,
    import_textstat,
)
from langchain.schema import AgentAction, AgentFinish, LLMResult


def import_flytekit():
    try:
        import flytekit  # noqa: F401
    except ImportError:
        raise ImportError(
            "To use the flyte callback manager you need to have the `flytekit` and"
            "`flytekitplugins-deck-standard` packages installed. Please install it with `pip install flytekit`"
        )


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
    resp: Dict[str, Any] = {}
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


class FlyteCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler):
    """This callback handler is designed specifically for usage within a Flyte task."""

    def __init__(self) -> None:
        """Initialize callback handler."""

        import_flytekit()
        self.pandas = import_pandas()
        spacy = import_spacy()
        super().__init__()

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

        self.records: Dict[str, Any] = {
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

        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer, MarkdownRenderer

        self.table_renderer = TableRenderer
        self.markdown_renderer = MarkdownRenderer

        self.deck = Deck(
            "Langchain Metrics",
            self.markdown_renderer().to_html("## Langchain Metrics"),
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""

        self.metrics["step"] += 1
        self.metrics["llm_starts"] += 1
        self.metrics["starts"] += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_llm_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        prompt_responses = []
        for prompt in prompts:
            prompt_responses.append(prompt)

        resp.update({"prompts": prompt_responses})

        self.deck.append(self.markdown_renderer().to_html("### LLM Start"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.metrics["step"] += 1
        self.metrics["llm_ends"] += 1
        self.metrics["ends"] += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_llm_end"})
        resp.update(flatten_dict(response.llm_output or {}))
        resp.update(self.metrics)

        all_complexity_metrics = []

        for generations in response.generations:
            for generation in generations:
                generation_resp = deepcopy(resp)
                generation_resp.update(flatten_dict(generation.dict()))
                generation_resp.update(
                    analyze_text(
                        generation.text,
                        nlp=self.nlp,
                    )
                )
                complexity_metrics: Dict[str, float] = generation_resp.pop("text_complexity_metrics")  # type: ignore  # noqa: E501
                all_complexity_metrics.append(complexity_metrics)

                dependency_tree = generation_resp["dependency_tree"]
                entities = generation_resp["entities"]

        self.deck.append(self.markdown_renderer().to_html("### LLM End"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp]))
            + "\n"
            + self.table_renderer().to_html(
                self.pandas.DataFrame.from_records(all_complexity_metrics)
            )
            + "\n"
            + dependency_tree
            + "\n"
            + entities
            + "\n"
        )

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
        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer
        import pandas as pd

        self.metrics["step"] += 1
        self.metrics["chain_starts"] += 1
        self.metrics["starts"] += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_chain_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        chain_input = ",".join([f"{k}={v}" for k, v in inputs.items()])
        input_resp = deepcopy(resp)
        input_resp["inputs"] = chain_input

        self.deck.append(self.markdown_renderer().to_html("### Chain Start"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([input_resp])) + "\n"
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer
        import pandas as pd

        self.metrics["step"] += 1
        self.metrics["chain_ends"] += 1
        self.metrics["ends"] += 1

        resp: Dict[str, Any] = {}
        chain_output = ",".join([f"{k}={v}" for k, v in outputs.items()])
        resp.update({"action": "on_chain_end", "outputs": chain_output})
        resp.update(self.metrics)

        self.deck.append(self.markdown_renderer().to_html("### Chain End"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

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
        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer
        import pandas as pd

        self.metrics["step"] += 1
        self.metrics["tool_starts"] += 1
        self.metrics["starts"] += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_tool_start", "input_str": input_str})
        resp.update(flatten_dict(serialized))
        resp.update(self.metrics)

        self.deck.append(self.markdown_renderer().to_html("### Tool Start"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer
        import pandas as pd

        self.metrics["step"] += 1
        self.metrics["tool_ends"] += 1
        self.metrics["ends"] += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_tool_end", "output": output})
        resp.update(self.metrics)

        self.deck.append(self.markdown_renderer().to_html("### Tool End"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

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
        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer
        import pandas as pd

        self.metrics["step"] += 1
        self.metrics["text_ctr"] += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_text", "text": text})
        resp.update(self.metrics)

        self.deck.append(self.markdown_renderer().to_html("### On Text"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer
        import pandas as pd

        self.metrics["step"] += 1
        self.metrics["agent_ends"] += 1
        self.metrics["ends"] += 1

        resp: Dict[str, Any] = {}
        resp.update(
            {
                "action": "on_agent_finish",
                "output": finish.return_values["output"],
                "log": finish.log,
            }
        )
        resp.update(self.metrics)

        self.deck.append(self.markdown_renderer().to_html("### Agent Finish"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        from flytekit import Deck
        from flytekitplugins.deck.renderer import TableRenderer
        import pandas as pd

        self.metrics["step"] += 1
        self.metrics["tool_starts"] += 1
        self.metrics["starts"] += 1

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

        self.deck.append(self.markdown_renderer().to_html("### Agent Action"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )
