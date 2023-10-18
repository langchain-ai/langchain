class CopyDocumentTransformer(BaseDocumentTransformer):
    def transform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return documents