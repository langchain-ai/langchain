"""Question answering over a graph."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
    DGRAPH_FIX_PROMPT,
    DGRAPH_QA_PROMPT,
    DQL_GENERATION_PROMPT,
    DQL_QUERY_EXAMPLE,
    DQL_QUERYSYNTAX_INJECT_STRING
)
from langchain.chains.llm import LLMChain
from langchain.graphs.arangodb_graph import ArangoGraph
from langchain.graphs.dgraph_graph import DGraph
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate

class DGraphQAChain(Chain):
    graph: DGraph = Field(exclude=True)
    dql_gen_chain: LLMChain
    dql_fix_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    
    # Class modifiers
    max_generation_attempts: int = 5
    dql_examples: str = ""
    use_generic_query_examples: bool = False
    # Flag to enable experimental DQL Syntax Injection
    experimental_dql_syntax_injection: bool = False 

    @property
    def input_keys(self) -> List[str]:
      """Input keys.

      :meta private:
      """
      return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
      """Output keys.

      :meta private:
      """
      return [self.output_key]
    
    def _get_dql_examples(self) -> str:
      """Get DQL Examples."""
      if self.use_generic_query_examples:
        return DQL_QUERY_EXAMPLE
      
      return self.dql_examples
    
    def _get_dql_syntax_notes(self) -> str:
      """Get DQL Syntax Notes."""
      if self.experimental_dql_syntax_injection:
        return DQL_QUERYSYNTAX_INJECT_STRING
      
      return ""
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        qa_prompt: BasePromptTemplate = DGRAPH_QA_PROMPT,
        dql_gen_prompt: BasePromptTemplate = DQL_GENERATION_PROMPT,
        dql_fix_prompt: BasePromptTemplate = DGRAPH_FIX_PROMPT,
        **kwargs: Any,
    ) -> DGraphQAChain:
      """Initialize from LLM."""
      qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
      dql_gen_chain = LLMChain(llm=llm, prompt=dql_gen_prompt)
      dql_fix_chain = LLMChain(llm=llm, prompt=dql_fix_prompt)
      return cls(
          qa_chain=qa_chain,
          dql_gen_chain=dql_gen_chain,
          dql_fix_chain=dql_fix_chain,
          **kwargs,
      )
      
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
      _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
      callbacks = _run_manager.get_child()
      user_input = inputs[self.input_key]
      dql_generation_output = self.dql_gen_chain.run(
        {
          "dgraph_schema": self.graph.get_schema(),
          "dql_examples": self._get_dql_examples(),
          "dql_syntax_notes": self._get_dql_syntax_notes(),
          "user_input": user_input,
        },
        callbacks=callbacks,
      )

      dql_query = ""
      dql_error = ""
      dql_result = None
      # Init to 1 since we already generated the first query
      generation_attempt = 1
      
      while (
        dql_result is None
        and generation_attempt <= self.max_generation_attempts
      ):
        dql_query = self._extract_dql_statement(dql_generation_output)
        _run_manager.on_text(
            f"AQL Query ({generation_attempt}):", verbose=self.verbose
        )
        _run_manager.on_text(
            dql_query, color="green", end="\n", verbose=self.verbose
        )
        
        # Execute the DQL query
        try:
          dql_result = self.graph.query(dql_query)
        except Exception as e:
          dql_error = e
          _run_manager.on_text(
              "DQL Query Execution Error: ", end="\n", verbose=self.verbose
          )
          _run_manager.on_text(
              dql_error, color="yellow", end="\n\n", verbose=self.verbose
          )
          
          # retry AQL generation
          dql_generation_output = self.dql_fix_chain.run(
            {
              "dgraph_schema": self.graph.get_schema(),
              "dql_query": dql_query,
              "user_input": user_input,
              "dql_error": dql_error,
              "dql_examples": self._get_dql_examples(),
            },
            callbacks=callbacks,
          )
          generation_attempt += 1
        
      if dql_result is None:
        message = f"""Failed to generate a valid DQL query after {generation_attempt} attempts.
                  Unable to execute DQL query because of the following error: {dql_error}"""
        raise ValueError(message)
      
      _run_manager.on_text("DQL Result:", end="\n", verbose=self.verbose)
      _run_manager.on_text(
        str(dql_result), color="green", end="\n", verbose=self.verbose
      )
      
      # Interpret DQL Result
      result = self.qa_chain(
        {
          "dgraph_schema": self.graph.get_schema(),
          "user_input": user_input,
          "dql_query": dql_query,
          "dql_result": dql_result,
        },
        callbacks=callbacks,
      )
      
      qa_result = {self.output_key: result[self.qa_chain.output_key]}
      return qa_result
      
      
    def _extract_dql_statement(self, text: str) -> str:
      """Extract DQL statement code from text using Regex."""
      pattern = r"```(.*?)```"
      # Find all matches in the input text
      matches = re.findall(pattern, text, re.DOTALL)
      return matches[0] if matches else text