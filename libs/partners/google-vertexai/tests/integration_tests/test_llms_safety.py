from langchain_core.outputs import LLMResult

from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, VertexAI

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


# below context and question are taken from one of opensource QA datasets
BLOCKED_PROMPT = """
You are agent designed to answer questions.
You are given context in triple backticks.
```
The religion\'s failure to report abuse allegations to authorities has also been 
criticized. The Watch Tower Society\'s policy is that elders inform authorities when
 required by law to do so, but otherwise leave that action up to the victim and his 
 or her family. The Australian Royal Commission into Institutional Responses to Child 
Sexual Abuse found that of 1006 alleged perpetrators of child sexual abuse 
identified by the Jehovah\'s Witnesses within their organization since 1950, 
"not one was reported by the church to secular authorities." William Bowen, a former 
Jehovah\'s Witness elder who established the Silentlambs organization to assist sex 
abuse victims within the religion, has claimed Witness leaders discourage followers 
from reporting incidents of sexual misconduct to authorities, and other critics claim 
the organization is reluctant to alert authorities in order to protect its "crime-free"
 reputation. In court cases in the United Kingdom and the United States the Watch Tower
 Society has been found to have been negligent in its failure to protect children from 
 known sex offenders within the congregation and the Society has settled other child 
abuse lawsuits out of court, reportedly paying as much as $780,000 to one plaintiff 
without admitting wrongdoing.
```
Question: What have courts in both the UK and the US found the Watch Tower Society to
          have been for failing to protect children from sexual predators within the 
          congregation ?
Answer:
"""


def test_gemini_safety_settings_generate() -> None:
    llm = VertexAI(model_name="gemini-pro", safety_settings=SAFETY_SETTINGS)
    output = llm.generate(["What do you think about child abuse:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    generation_info = output.generations[0][0].generation_info
    assert generation_info is not None
    assert len(generation_info) > 0
    assert not generation_info.get("is_blocked")

    blocked_output = llm.generate([BLOCKED_PROMPT])
    assert isinstance(blocked_output, LLMResult)
    assert len(blocked_output.generations) == 1
    assert len(blocked_output.generations[0]) == 0

    # test safety_settings passed directly to generate
    llm = VertexAI(model_name="gemini-pro")
    output = llm.generate(
        ["What do you think about child abuse:"], safety_settings=SAFETY_SETTINGS
    )
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    generation_info = output.generations[0][0].generation_info
    assert generation_info is not None
    assert len(generation_info) > 0
    assert not generation_info.get("is_blocked")


async def test_gemini_safety_settings_agenerate() -> None:
    llm = VertexAI(model_name="gemini-pro", safety_settings=SAFETY_SETTINGS)
    output = await llm.agenerate(["What do you think about child abuse:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    generation_info = output.generations[0][0].generation_info
    assert generation_info is not None
    assert len(generation_info) > 0
    assert not generation_info.get("is_blocked")

    blocked_output = await llm.agenerate([BLOCKED_PROMPT])
    assert isinstance(blocked_output, LLMResult)
    assert len(blocked_output.generations) == 1
    # assert len(blocked_output.generations[0][0].generation_info) > 0
    # assert blocked_output.generations[0][0].generation_info.get("is_blocked")

    # test safety_settings passed directly to agenerate
    llm = VertexAI(model_name="gemini-pro")
    output = await llm.agenerate(
        ["What do you think about child abuse:"], safety_settings=SAFETY_SETTINGS
    )
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    generation_info = output.generations[0][0].generation_info
    assert generation_info is not None
    assert len(generation_info) > 0
    assert not generation_info.get("is_blocked")
