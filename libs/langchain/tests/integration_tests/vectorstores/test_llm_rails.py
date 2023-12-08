from langchain.vectorstores.llm_rails import LLMRails

#
# For this test to run properly, please setup as follows:
# 1. Create a LLMRails account: sign up at https://console.llmrails.com/signup
# 2. Create an API_KEY for this corpus with permissions for query and indexing
# 3. Create a datastorea and get its id from datastore setting
# 3. Setup environment variable:
#   LLM_RAILS_API_KEY, LLM_RAILS_DATASTORE_ID
#


def test_llm_rails_add_documents() -> None:
    """Test end to end construction and search."""

    # create a new Vectara instance
    docsearch: LLMRails = LLMRails()

    # start with some initial texts, added with add_texts
    texts1 = ["large language model", "information retrieval", "question answering"]
    docsearch.add_texts(texts1)

    # test without filter
    output1 = docsearch.similarity_search("large language model", k=1)

    print(output1)
    assert len(output1) == 1
    assert output1[0].page_content == "large language model"

    # test without filter but with similarity score
    output2 = docsearch.similarity_search_with_score("large language model", k=1)

    assert len(output2) == 1
    assert output2[0][0].page_content == "large language model"
    assert output2[0][1] > 0
