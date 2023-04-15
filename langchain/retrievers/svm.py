"""SMV Retriever.
Largely based on
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb"""

import numpy as np
from sklearn import svm
from typing import Any, List
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document

def create_index(contexts: List[str], 
                 embeddings: Embeddings) -> "SVMRetriever":
    
    return np.array([embeddings.embed_query(split) for split in contexts])

class SVMRetriever(BaseRetriever, BaseModel):

    embeddings: Embeddings
    index: Any
    texts: List[str]
    k: int = 4

    class Config:

        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def from_texts(self, texts: List[str], embeddings) -> None:

        self.texts = texts
        self.index = create_index(texts, embeddings)

    def get_relevant_documents(self, query: str) -> List[Document]:
        
        query = np.array(self.embeddings.embed_query(query))
        x = np.concatenate([query[None, ...], self.index])
        y = np.zeros(x.shape[0])
        y[0] = 1
        
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        clf.fit(x, y)
        
        similarities = clf.decision_function(x)
        sorted_ix = np.argsort(-similarities)
        
        top_k_results = []
        for row in sorted_ix[1:self.k+1]:
            top_k_results.append(Document(page_content=self.texts[row]))
        return top_k_results