import langchain
from langchain.audio_models import AudioBanana
from langchain.cache import SQLiteCache
from langchain.chains import AudioChain, LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# TODO: Turn this into a notebook

audio_cache = SQLiteCache(database_path=".langchain.db")

langchain.llm_cache = audio_cache

# Uses "lucataco/serverless-template-whisper-largev2:V1" or "banana/Whisper - Baseon" Banana dev # noqa: E501

audio_model = AudioBanana(model_key="[YOUR MODEL KEY]", max_chars=20000)

llm = OpenAI(temperature=0.7)
template = """Speech: {transcript}

Write a short 3 sentence summary of the speech.

Summary:"""

prompt_template = PromptTemplate(input_variables=["transcript"], template=template)
summary_chain = LLMChain(llm=llm, prompt=prompt_template)

audio_chain = AudioChain(audio_model=audio_model, output_key="transcript")

# AudioChain takes in an mp3 file string as input and returns a transcript

speech_summary_chain = SimpleSequentialChain(
    chains=[audio_chain, summary_chain], verbose=True
)

# wget https://dl.dropboxusercontent.com/s/pw27877dknh82tk/Audio%20-%20Martin%20Luther%20King%20-%20I%20Have%20A%20Dream.mp3?dl=0 -O ihaveadream.mp3 # noqa: E501

speech_summary_chain.run("ihaveadream.mp3")
