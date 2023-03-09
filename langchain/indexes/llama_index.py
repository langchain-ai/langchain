from langchain.schema import BaseIndex, Document
from pydantic import BaseModel
from typing import Any, List


class LlamaIndex(BaseIndex, BaseModel):

    index: Any

    def get_relevant_texts(self, query: str, **kwargs: Any) -> List[Document]:
        response = self.index.query(query, response_mode="no_text")
        return [Document(page_content=r.source_text) for r in response.source_nodes]
