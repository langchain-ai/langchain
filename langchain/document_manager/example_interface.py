from in_memory import InMemoryDocumentManager
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

_TEST_DOCUMENTS = [Document(page_content='This is a test document.'),
                   Document(page_content='This is another test document.'),
                   Document(page_content='This is a third test document.')]


def main():
    # Create a document manager
    document_manager = InMemoryDocumentManager(CharacterTextSplitter(separator=" "))

    # Add some documents.
    ops = document_manager.add(_TEST_DOCUMENTS, ['1', '2', '3'])
    print(ops)
    print('-'*100)
    # Apply change to vector store.
    # VectorStore.apply_operations(ops)

    # Update documents.
    ops = document_manager.update([Document(page_content='This is a modified test document.')], ['1'])
    print(ops)
    print('-'*100)
    # Apply change to vector store.
    # VectorStore.apply_operations(ops)

    # Replacing existing documents.
    ops = document_manager.update_truncate([Document(page_content='This is the final test document.'),
                                            Document(page_content='This is another final test document')], 
                                           ['1', '2'])
    print(ops)
    # Apply change to vector store.
    # VectorStore.apply_operations(ops)




if __name__ == '__main__':
    main()


