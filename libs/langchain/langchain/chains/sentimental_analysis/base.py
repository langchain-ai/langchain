from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union
import re 

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sentimental_analysis.prompt import SENTIMENT_PROMPT
from langchain.schema.language_model import BaseLanguageModel
from langchain.pydantic_v1 import Field,root_validator,Extra 

logger = logging.getLogger(__name__)

class SentimentAnalysisChain(Chain):
    """Chain that performs sentiment analysis on text.

    Example:
        .. code-block:: python

            from langchain.chains import SentimentAnalysis
            from langchain.llms import OpenAI
            sentiment_chain = SentimentAnalysis.from_llm(OpenAI())
    """
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    input_key: str = "text"  #: :meta private:
    output_key: str = "sentiment"  #: :meta private:
    custom_prompt_template: Optional[str] = SENTIMENT_PROMPT.template  # Use the updated prompt template
    include_score: bool = True  #: Option to include sentiment score in output
    output_format: str = "json"  #: Output format for sentiment analysis results (default is JSON)
    sentiment_label_mapping: Optional[Dict[str,str]] = Field(default_factory=dict)  #: Custom sentiment label mapping
    batch_processing: bool = False  #: Enable batch processing of multiple text inputs

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating a SentimentAnalysis chain with an llm is deprecated. "
                "Please instantiate with llm_chain or using the from_llm class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", SENTIMENT_PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[Dict[str, Any], str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key], verbose=self.verbose)

        # Check if batch processing is enabled and input is a list
        if self.batch_processing and isinstance(inputs[self.input_key], List):
            results = [self._process_single_input(text_input,_run_manager) for text_input in inputs[self.input_key]]
            return results
        else:
            return self._process_single_input(inputs[self.input_key], _run_manager)

    def _process_single_input(
        self,
        text: str,
        run_manager: CallbackManagerForChainRun,
    ) -> Union[Dict[str, Any], str]:
        run_manager.on_text(text, verbose=self.verbose)

        t = self.llm_chain.predict(
            question=text, callbacks=run_manager.get_child()
        )
        run_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()
        
        # Process LLM results and extract sentiment using regular expressions
        sentiment = self.process_llm_results(t)

        if self.verbose:
            run_manager.on_text("\nSentiment: ", verbose=self.verbose)
            run_manager.on_text(
                str(sentiment), color="yellow", verbose=self.verbose
            )

        if self.output_format.lower() == "json":
            # Return the sentiment analysis results as a JSON dictionary
            return {self.output_key: sentiment}
        elif self.output_format.lower() == "text":
            # Return the sentiment analysis results as plain text
            return f"Sentiment Label: {sentiment['sentiment_label']}\nSentiment Score: {sentiment['sentiment_score']}"
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def process_llm_results(self, text: str) -> dict:
        sentiment_pattern = re.compile(r'Sentiment: (.+?) \(Score: ([\d.]+)\)')
        match = sentiment_pattern.search(text)
        if match:
            sentiment_label = match.group(1).strip()
            sentiment_score = float(match.group(2).strip())
            
            # Apply custom sentiment label mapping
            sentiment_label = self.sentiment_label_mapping.get(sentiment_label, sentiment_label)
            
            if self.include_score:
                return {"sentiment_label": sentiment_label, "sentiment_score": sentiment_score}
            else:
                return {"sentiment_label": sentiment_label}
        return {}

    @property
    def _chain_type(self) -> str:
        return "SentimentAnalysisChain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        custom_prompt_template: Optional[str] = None,
        include_score: bool = True,
        output_format: str = "json",
        sentiment_label_mapping: Optional[Dict] = Field(default_factory=dict),
        batch_processing: bool = False,
        **kwargs: Any,
    ) -> SentimentAnalysisChain:
        
        sentiment_prompt = custom_prompt_template if custom_prompt_template is not None else SENTIMENT_PROMPT
        llm_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
        return cls(
            llm_chain=llm_chain,
            custom_prompt_template=custom_prompt_template,
            include_score=include_score,
            output_format=output_format,
            sentiment_label_mapping=sentiment_label_mapping,
            batch_processing=batch_processing,
            **kwargs
        )

