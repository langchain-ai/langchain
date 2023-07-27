"""FlyteKit callback handler."""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.utils import (
    BaseMetadataCallbackHandler,
    flatten_dict,
    import_pandas,
    import_spacy,
    import_textstat,
)
from langchain.schema import AgentAction, AgentFinish, LLMResult

if TYPE_CHECKING:
    import flytekit
    from flytekitplugins.deck import renderer

logger = logging.getLogger(__name__)


def import_flytekit() -> Tuple[flytekit, renderer]:
    """Import flytekit and flytekitplugins-deck-standard."""
    try:
        import flytekit  # noqa: F401
        from flytekitplugins.deck import renderer  # noqa: F401
    except ImportError:
        raise ImportError(
            "To use the flyte callback manager you need"
            "to have the `flytekit` and `flytekitplugins-deck-standard`"
            "packages installed. Please install them with `pip install flytekit`"
            "and `pip install flytekitplugins-deck-standard`."
        )
    return flytekit, renderer


def analyze_text(
    text: str,
    nlp: Any = None,
    textstat: Any = None,
) -> dict:
    """Analyze text using textstat and spacy.

    Parameters:
        text (str): The text to analyze.
        nlp (spacy.lang): The spacy language model to use for visualization.

    Returns:
        (dict): A dictionary containing the complexity metrics and visualization
            files serialized to HTML string.
    """
    resp: Dict[str, Any] = {}
    if textstat is not None:
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
        spacy = import_spacy()
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
    """This callback handler that is used within a Flyte task."""

    def __init__(self) -> None:
        """Initialize callback handler."""
        flytekit, renderer = import_flytekit()
        self.pandas = import_pandas()

        self.textstat = None
        try:
            self.textstat = import_textstat()
        except ImportError:
            logger.warning(
                "Textstat library is not installed. \
                It may result in the inability to log \
                certain metrics that can be captured with Textstat."
            )

        spacy = None
        try:
            spacy = import_spacy()
        except ImportError:
            logger.warning(
                "Spacy library is not installed. \
                It may result in the inability to log \
                certain metrics that can be captured with Spacy."
            )

        super().__init__()

        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "FlyteCallbackHandler uses spacy's en_core_web_sm model"
                    " for certain metrics. To download,"
                    " run the following command in your terminal:"
                    " `python -m spacy download en_core_web_sm`"
                )

        self.table_renderer = renderer.TableRenderer
        self.markdown_renderer = renderer.MarkdownRenderer

        self.deck = flytekit.Deck(
            "LangChain Metrics",
            self.markdown_renderer().to_html("## LangChain Metrics"),
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""

        self.step += 1
        self.llm_starts += 1
        self.starts += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_llm_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())

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

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.step += 1
        self.llm_ends += 1
        self.ends += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_llm_end"})
        resp.update(flatten_dict(response.llm_output or {}))
        resp.update(self.get_custom_callback_meta())

        self.deck.append(self.markdown_renderer().to_html("### LLM End"))
        self.deck.append(self.table_renderer().to_html(self.pandas.DataFrame([resp])))

        for generations in response.generations:
            for generation in generations:
                generation_resp = deepcopy(resp)
                generation_resp.update(flatten_dict(generation.dict()))
                if self.nlp or self.textstat:
                    generation_resp.update(
                        analyze_text(
                            generation.text, nlp=self.nlp, textstat=self.textstat
                        )
                    )

                    complexity_metrics: Dict[str, float] = generation_resp.pop("text_complexity_metrics")  # type: ignore  # noqa: E501
                    self.deck.append(
                        self.markdown_renderer().to_html("#### Text Complexity Metrics")
                    )
                    self.deck.append(
                        self.table_renderer().to_html(
                            self.pandas.DataFrame([complexity_metrics])
                        )
                        + "\n"
                    )

                    dependency_tree = generation_resp["dependency_tree"]
                    self.deck.append(
                        self.markdown_renderer().to_html("#### Dependency Tree")
                    )
                    self.deck.append(dependency_tree)

                    entities = generation_resp["entities"]
                    self.deck.append(self.markdown_renderer().to_html("#### Entities"))
                    self.deck.append(entities)
                else:
                    self.deck.append(
                        self.markdown_renderer().to_html("#### Generated Response")
                    )
                    self.deck.append(self.markdown_renderer().to_html(generation.text))

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
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

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_chain_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())

        chain_input = ",".join([f"{k}={v}" for k, v in inputs.items()])
        input_resp = deepcopy(resp)
        input_resp["inputs"] = chain_input

        self.deck.append(self.markdown_renderer().to_html("### Chain Start"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([input_resp])) + "\n"
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.step += 1
        self.chain_ends += 1
        self.ends += 1

        resp: Dict[str, Any] = {}
        chain_output = ",".join([f"{k}={v}" for k, v in outputs.items()])
        resp.update({"action": "on_chain_end", "outputs": chain_output})
        resp.update(self.get_custom_callback_meta())

        self.deck.append(self.markdown_renderer().to_html("### Chain End"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
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

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_tool_start", "input_str": input_str})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())

        self.deck.append(self.markdown_renderer().to_html("### Tool Start"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.step += 1
        self.tool_ends += 1
        self.ends += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_tool_end", "output": output})
        resp.update(self.get_custom_callback_meta())

        self.deck.append(self.markdown_renderer().to_html("### Tool End"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        self.step += 1
        self.errors += 1

    def on_text(self, text: str, **kwargs: Any) -> None:
        """
        Run when agent is ending.
        """
        self.step += 1
        self.text_ctr += 1

        resp: Dict[str, Any] = {}
        resp.update({"action": "on_text", "text": text})
        resp.update(self.get_custom_callback_meta())

        self.deck.append(self.markdown_renderer().to_html("### On Text"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        self.step += 1
        self.agent_ends += 1
        self.ends += 1

        resp: Dict[str, Any] = {}
        resp.update(
            {
                "action": "on_agent_finish",
                "output": finish.return_values["output"],
                "log": finish.log,
            }
        )
        resp.update(self.get_custom_callback_meta())

        self.deck.append(self.markdown_renderer().to_html("### Agent Finish"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.step += 1
        self.tool_starts += 1
        self.starts += 1

        resp: Dict[str, Any] = {}
        resp.update(
            {
                "action": "on_agent_action",
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
        )
        resp.update(self.get_custom_callback_meta())

        self.deck.append(self.markdown_renderer().to_html("### Agent Action"))
        self.deck.append(
            self.table_renderer().to_html(self.pandas.DataFrame([resp])) + "\n"
        )
