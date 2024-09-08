from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from libs.community.langchain_community.chains.reflection.chain import ReflectionOutputTagExtractorChain

# Create the LLM
llm = OllamaLLM(model="reflection")

# Create the prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question:\n\n{question}"
)

# Create the OutputTagExtractorChain
chain = ReflectionOutputTagExtractorChain.from_llm(llm=llm, prompt=prompt)

# Use the chain
response = chain.run("Who was the first person to walk on the moon?")
print(response)