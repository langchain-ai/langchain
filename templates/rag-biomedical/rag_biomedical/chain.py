from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
# Embed a single document as a test
vectorstore = Chroma(persist_directory="./vectorstore", 
                     collection_name="rag-biomedical",
                     embedding_function=OpenAIEmbeddings())



# Provide context about the metadata
metadata_field_info = [
    
    AttributeInfo(
        name="NCTId",
        description="The unique identifier assigned to each clinical trial when registered on ClinicalTrials.gov. ",
        type="string",
    ),
    AttributeInfo(
        name="StudyType",
        description="The nature of the study, indicating whether participants receive specific interventions or are merely observed for specific outcomes.",
        type="string",
    ),
    AttributeInfo(
        name="PhaseList",
        description="This pertains to the phase of the study in drug trials.",
        type="string",
    )
]

# Overall context for the data
document_content_description = "Information about clinical trial on ClinicalTrials.gov"

# LLM
llm = OpenAI(temperature=0,)

# Retriever
retriever_self_query = SelfQueryRetriever.from_llm(
    llm, 
    vectorstore, 
    document_content_description, 
    metadata_field_info, 
    verbose=True
)

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# RAG chain
chain = (
    RunnableParallel({"context": retriever_self_query, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
