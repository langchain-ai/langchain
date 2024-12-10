from typing import Any, List, Optional

from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document


class LindormParentDocumentRetriever(ParentDocumentRetriever):
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can be provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.

            kwargs: Additional keyword arguments passed to the parent document
                routing_field: which field in metadata to use as the document's routing
                              key
                tag: which field in metadata to use as the document's identity
                metadata: global metadata to override parent document's metadata

                {
                    "routing_field": "split_setting",
                    "tag": "source",
                    "metadata":{
                        "split_setting": "10"
                     }
                }

        """
        routing_field = kwargs.pop("routing_field", "split_setting")
        tag = kwargs.pop("tag", "source")
        metadata = kwargs.pop("metadata", {})
        routing = metadata.get(routing_field, "")
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        parent_docs = []
        child_docs = []
        for i, doc in enumerate(documents):
            doc.metadata.update(metadata)
            _id = f"{doc.metadata[tag]}_{routing}_parent_{i}"
            doc.id = _id
            sub_docs = self.child_splitter.split_documents([doc])
            for j, _doc in enumerate(sub_docs):
                _doc.metadata[self.id_key] = _id
                _doc.id = f"{_id}_child_{j}"
            child_docs.extend(sub_docs)
            parent_docs.append((_id, doc))
        child_ids = [d.id for d in child_docs]
        self.vectorstore.add_documents(child_docs, ids=child_ids, **kwargs)
        self.docstore.mset(parent_docs)
