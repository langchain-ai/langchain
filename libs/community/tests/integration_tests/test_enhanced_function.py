from langchain_community.chains.enhanced_function import EnhancedFunctionChain
from langchain.llms import OpenAI

def test_enhanced_function_chain():
    llm = OpenAI(temperature=0)
    chain = EnhancedFunctionChain(llm=llm)

    # Test math query
    result = chain.run(query="What is the square root of 144?")
    assert "12" in result["result"].lower()

    # Test non-math query
    result = chain.run(query="What is the capital of France?")
    assert "paris" in result["result"].lower()

    # Test explicit math flag
    chain.contains_math = True
    result = chain.run(query="What is 2 + 2?")
    assert "4" in result["result"]
