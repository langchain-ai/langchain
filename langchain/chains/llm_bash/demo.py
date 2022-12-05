from dotenv import load_dotenv

load_dotenv()

from langchain.chains.llm_bash.base import LLMBashChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain = LLMBashChain(llm=llm, verbose=True)

bash_chain.run(text)
