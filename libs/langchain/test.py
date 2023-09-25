from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.evaluation.comparison.llm_as_a_judge import LLMAsAJudgePairwiseEvalChain

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4")

evaluator = LLMAsAJudgePairwiseEvalChain.from_llm(llm=llm)

e = evaluator.evaluate_string_pairs(
    prediction="The chemical formula for water is H2O, which means there are two hydrogen atoms and one oxygen atom",
    prediction_b="The chemical formula for water is H2O.",
    input="What is the chemical formula for water?",
)

print(e)
