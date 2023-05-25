from langchain.base_language import BaseLanguageModel
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.document_loaders import DialogueLoader
from langchain.chains.dialogue_answering.prompts import (
    DIALOGUE_PREFIX,
    DIALOGUE_SUFFIX,
    SUMMARY_PROMPT
)


class DialogueWithSharedMemoryChains:
    zero_shot_react_llm: BaseLanguageModel = None
    ask_llm: BaseLanguageModel = None
    embeddings: HuggingFaceEmbeddings = None
    embedding_model: str = None
    vector_search_top_k: int = 6
    dialogue_path: str = None
    dialogue_loader: DialogueLoader = None
    device: str = None

    def __init__(self, zero_shot_react_llm: BaseLanguageModel = None, ask_llm: BaseLanguageModel = None,
                 params: dict = None):
        self.zero_shot_react_llm = zero_shot_react_llm
        self.ask_llm = ask_llm
        params = params or {}
        self.embedding_model = params.get('embedding_model', 'GanymedeNil/text2vec-large-chinese')
        self.vector_search_top_k = params.get('vector_search_top_k', 6)
        self.dialogue_path = params.get('dialogue_path', '')
        self.device = 'cuda' if params.get('use_cuda', False) else 'cpu'

        self.dialogue_loader = DialogueLoader(self.dialogue_path)
        self._init_cfg()
        self._init_state_of_history()
        self.memory_chain, self.memory = self._agents_answer()
        self.agent_chain = self._create_agent_chain()

    def _init_cfg(self):
        model_kwargs = {
            'device': self.device
        }
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs=model_kwargs)

    def _init_state_of_history(self):
        documents = self.dialogue_loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=3, chunk_overlap=1)
        texts = text_splitter.split_documents(documents)
        docsearch = Chroma.from_documents(texts, self.embeddings, collection_name="state-of-history")
        self.state_of_history = RetrievalQA.from_chain_type(llm=self.ask_llm, chain_type="stuff",
                                                            retriever=docsearch.as_retriever())

    def _agents_answer(self):

        memory = ConversationBufferMemory(memory_key="chat_history")
        readonly_memory = ReadOnlySharedMemory(memory=memory)
        memory_chain = LLMChain(
            llm=self.ask_llm,
            prompt=SUMMARY_PROMPT,
            verbose=True,
            memory=readonly_memory,  # use the read-only memory to prevent the tool from modifying the memory
        )
        return memory_chain, memory

    def _create_agent_chain(self):
        dialogue_participants = self.dialogue_loader.dialogue.participants_to_export()
        tools = [
            Tool(
                name="State of Dialogue History System",
                func=self.state_of_history.run,
                description=f"Dialogue with {dialogue_participants} - The answers in this section are very useful "
                            f"when searching for chat content between {dialogue_participants}. Input should be a "
                            f"complete question. "
            ),
            Tool(
                name="Summary",
                func=self.memory_chain.run,
                description="useful for when you summarize a conversation. The input to this tool should be a string, "
                            "representing who will read this summary. "
            )
        ]

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=DIALOGUE_PREFIX,
            suffix=DIALOGUE_SUFFIX,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )

        llm_chain = LLMChain(llm=self.zero_shot_react_llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=self.memory)

        return agent_chain
