"""Test conversational router chain and memory."""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pytest
import yaml
from langchain.chains.llm_router_chain.base import RouterChain
from langchain import LLMChain, PromptTemplate
from tests.unit_tests.llms.fake_llm import FakeLLM

LLM_MAP_CONFIG = '''
    models:
      - Space:
          qa_maker:
            - How far is the earth from the moon?
            - What's the temperature of the sun?
            - How does the air smell in venus?
          template: |
            Assume that your Elon musk and are very concerned about future of human civilization beyond Earth. 

            Answer the following question keeping this in mind and provide answers that help in clarifying how 
            would humans survive as an interplanetary species. If the question is not relevant then say "I don't know" and do not make up any answer.
            Question related to space and how humans could survive:         
            {question}
          input_vars:
            - question
      - Architecture:
          qa_maker:
            - What's the best way to do sampling for statistical analysis?
            - Which technologies would make most sense for distributed work?
          template: |
            Assume the role of a software architect who's really experienced in dealing with and scaling large scale distributed systems. 
            Answer the questions specifically on software design problems as indicated below. If the question is not relevant then say "I don't know" and do not make up any answer. 

            Question related to distributed systems and large scale software design
            {question}

            Please also include references in your answers to popular websites where more we can get more context.
          input_vars:
            - question
      - Biotechnology:
          qa_maker:
            - What's the best way to tap into genetic memory?
          template: |
            Assume the role of a genetic expert who has unlocked the secrets of our genetic make up and is able to provide clear answers to questions below.
            Optimize for answers that provide directions for improving current problems around genetic defects and how we can overcome them.  If the question is not relevant then say "I don't know" and do not make up any answer.

            Question related to bio technology and related use cases.                 
            {question}
          input_vars:
            - question
    '''
LLM = FakeLLM()


class RouterConfig:
    def __init__(self, llm):
        self.chain_map = {}
        chroma_client = chromadb.Client()
        sentence_transformer_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.router_coll = chroma_client.create_collection(name='router', embedding_function=sentence_transformer_ef)
        content = yaml.safe_load(LLM_MAP_CONFIG)
        for model in content.get('models'):
            for m_name, m_content in model.items():
                m_name = m_name.lower()
                self.router_coll.add(ids=[str(x) for x in range(len(m_content.get('qa_maker')))],
                                     documents=m_content.get('qa_maker'),
                                     metadatas=[{'classification': m_name} for x in
                                                range(len(m_content.get('qa_maker')))])
                self.chain_map[m_name] = LLMChain(llm=llm, prompt=PromptTemplate(template=m_content.get('template'),
                                                                                 input_variables=m_content.get(
                                                                                     'input_vars')))

    def get_chains(self):
        return self.chain_map

    def get_embedding(self):
        return self.router_coll


@pytest.fixture()
def router_config() -> RouterChain:
    config = RouterConfig(llm=LLM)

    def vector_lookup(query):
        x = config.get_embedding().query(query_texts=query, n_results=3)
        return x['metadatas'][0][0].get('classification'), x['distances'][0][0]

    return RouterChain(chains=config.get_chains(),
                       vector_lookup_fn=vector_lookup)


def test_vector_selection_and_routing(router_config: RouterChain) -> None:
    """Test that the vector search has a hit and is able to pick a destination chain."""
    output = router_config.run("How far is the moon from the earth?")
    assert output['chain'] == 'space'
    assert output['output'] == 'foo'


def test_vector_memory_state(router_config: RouterChain) -> None:
    """Test that the conversational router chain is able to maintain historical context"""
    output1 = router_config.run("How far is the moon from th earth?")
    output2 = router_config.run("Tell me more!")
    assert output1['chain'] == 'space'
    assert output1['output'] == 'foo'
    assert output2['chain'] == 'space'
    assert output2['output'] == 'foo'


def test_memory_context_switch_across_chains(router_config: RouterChain) -> None:
    """Test that the history is maintained even if the conversation spans across chains"""
    output1 = router_config.run(input="How far is the moon from th earth?")
    output2 = router_config.run(input="How do you scale databases for large scale software development?")
    output3 = router_config.run(input="How do you solve genetic problems through AI/ML?")
    output4 = router_config.run(input="Tell me more about it!")
    assert output1['chain'] == 'space'
    assert output1['output'] == 'foo'
    assert output2['chain'] == 'architecture'
    assert output3['chain'] == 'biotechnology'
    assert output4['chain'] == 'biotechnology'
