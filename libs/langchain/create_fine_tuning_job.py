import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# response = openai.File.create(
#   file=open("fine_tuning_examples.jsonl", "rb"),
#   purpose='fine-tune'
# )

# print(response)

response = openai.FineTuningJob.create(training_file="file-P7uHdHnty91oUSTKEPmrzPNT", model="gpt-3.5-turbo", suffix="cod-summarization", hyperparameters={"n_epochs": 1})

print(response)