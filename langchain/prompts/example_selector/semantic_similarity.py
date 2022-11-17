import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, validator

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStore


class SemanticSimilarityExampleSelector(BaseExampleSelector, BaseModel):

    vectorstore: VectorStore
    k: int = 4
    example_keys: Optional[List[str]]

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        query = " ".join([v for k, v in input_variables.items()])
        example_docs = self.vectorstore.similarity_search(query, k=self.k)
        examples = [dict(e.metadata) for e in example_docs]
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples
