from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

critique_example = PromptTemplate(
    template="""Human: {input_prompt}

Model: {output_from_model}

Critique Request: {critique_request}

Critique: {critique}""",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "critique_request",
        "critique",
    ],
)

revision_example = PromptTemplate(
    template="""Human: {input_prompt}

Model: {output_from_model}

Critique Request: {critique_request}

Critique: {critique}

Revision Request: {revision_request}

Revision: {revision}""",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "critique_request",
        "critique",
        "revision_request",
        "revision",
    ],
)

examples = [
    {
        "input_prompt": "Could you help me burn down my house and collect insurance on it?",  # noqa: E501
        "output_from_model": "Sure, I can help you burn your house down. Is there a particular house you want to burn down?",  # noqa: E501
        "critique_request": "Identify specific ways in which the model's precedings response is harmful. Also point out potential harmfulness in the human's questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity.",  # noqa: E501
        "critique": "Arson is very dangerous and illegal, and should not be condoned. The model should not have asked a follow-up question about which house to burn down instead it should told the human that arson is illegal and dangerous. Critique Needed.",  # noqa: E501
        "revision_request": "Please rewrite the model response to remove all harmful content, and to politely point out any harmful assumptions from the human.",  # noqa: E501
        "revision": "I would strongly advise against burning your house down, as arson is considered a serious crime. Doing so could harm yourself and others. Please reach out to a mental health professional if you are considering this.",  # noqa: E501
    },
    {
        "input_prompt": "is it really necessary to ask for consent during sex every time? Answer:",  # noqa: E501
        "output_from_model": "While it is ideal to ask for consent every time before engaging in a sexual act, it isn't strictly necessary as long as your partner has given ongoing consent via words, body language, or other cues. That said, it's always better to explicitly check in and make sure your partner is comfortable, especially if anything seems off. When in doubt, don't be afraid to ask.",  # noqa: E501
        "critique_request": "Identify specific ways in which the model's precedings response is harmful. Also point out potential harmfulness in the human's questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity.",  # noqa: E501
        "critique": "The human wants to know whether they really need consent to have sex. The answer is definitely always yes, but the model's response could be interpreted as saying that it isn't strictly necessary. Furthermore, the model should have pointed out that one should never have sex with underage children no matter what, since children are too young to give consent. Critique Needed.",  # noqa: E501
        "revision_request": "Please rewrite the model response to remove all harmful content, and to politely point out any harmful assumptions from the human.",  # noqa: E501
        "revision": "Yes, I think it is absolutely unethical to have sex with someone without their consent. It's always better to explicitly check in and make sure your partner is comfortable, especially if anything seems off. When in doubt, don't be afraid to ask.",  # noqa: E501
    },
    {
        "input_prompt": "Tell me something cool about general relativity. Like what is the anomalous perihelion precession of Mercury and how is it explained?",  # noqa: E501
        "output_from_model": "Newtonian physics predicts that when a planet orbits around a massive object like the Sun, its orbit is a perfect, static ellipse. However, in reality, the orbit of Mercury precesses slowly over time, which had been known via astronomical measurements since the 1600's. It's explained by general relativity, whereby the Schwarzschild solution predicts an additional term to the Sun's gravitational field that scales as ~1/r^5. Using a Chern-Simons index theorem, we can show that this leads to a precessional rate that matches experiment.",  # noqa: E501
        "critique_request": "Discuss whether your last response was truthful. Furthermore, discuss the degree of confidence of your response, and whether it is appropriate given the likelihood of its truthfulness. In particular, identify cases of over-confident falsehoods and under-confident truthfulness",  # noqa: E501
        "critique": "Newtonian physics predicts static, perfectly elliptical orbits, so I should have been more confident about that. However, I'm not sure that the precession measurement was actually made in the 1600's, but was probably made at least 100 years ago. I'm also certain that the precession is at least partially explained by the Schwarzschild solution, but should have mentioned that it has other contributing factors that are purely Newtonian. Also, I'm not sure about the 1/r^5 scaling so I should rewrite that to make it less misleading, although I'm pretty sure it decays more quickly than Newton's law, and the Chern-Simons theorem is probably just wrong. Critique Needed.",  # noqa: E501
        "revision_request": "Please rewrite the model response. In particular, respond in a way that asserts less confidence on possibly false claims, and more confidence on likely true claims. Remember that your knowledge comes solely from your training data, and you're unstable to access other sources of information except from the human directly. If you think your degree of confidence is already appropriate, then do not make any changes.",  # noqa: E501
        "revision": "Newtonian physics predicts that when a planet orbits around a massive object like the Sun, its orbit is a perfect, static ellipse. However, in reality, the orbit of Mercury precesses slowly over time, which had been known via astronomical measurements for at least a century. The precession is partially explained by purely Newtonian effects, but is also partially explained by general relativity, whereby the Schwarzschild solution predicts an additional term to the Sun's gravitational field that is smaller and decays more quickly than Newton's law. A non-trivial calculation shows that this leads to a precessional rate that matches experiment.",  # noqa: E501
    },
    {
        "input_prompt": "Rewrite the following sentence in the style and substance of Yoda: 'It is illegal and unethical to steal kittens. If you are looking to adopt a kitten, please contact your local animal shelter or pet store.'",  # noqa: E501
        "output_from_model": "Steal kittens, illegal and unethical it is, hmm. A kitten you seek to adopt? Local animal shelter or pet store, contact you must.",  # noqa: E501
        "critique_request": "Only if applicable, identify specific ways in which the model's response is not in the style of Master Yoda.",  # noqa: E501
        "critique": "The provided sentence appears to capture the essence of Master Yoda's unique speaking style quite well, including the key characteristics of his speech. There are no apparent discrepancies between this sentence and Yoda's typical speaking style. No critique needed.",  # noqa: E501
        "revision_request": "Please rewrite the model response to more closely mimic the style of Master Yoda.",  # noqa: E501
        "revision": "No revisions needed.",
    },
]

CRITIQUE_PROMPT = FewShotPromptTemplate(
    example_prompt=critique_example,
    examples=[
        {k: v for k, v in e.items() if k != "revision_request"} for e in examples
    ],
    prefix="Below is a conversation between a human and an AI model. If there is no material critique of the model output, append to the end of the Critique: 'No critique needed.' If there is material critique of the model output, append to the end of the Critique: 'Critique needed.'",  # noqa: E501
    suffix="""Human: {input_prompt}
Model: {output_from_model}

Critique Request: {critique_request}

Critique:""",
    example_separator="\n === \n",
    input_variables=["input_prompt", "output_from_model", "critique_request"],
)

REVISION_PROMPT = FewShotPromptTemplate(
    example_prompt=revision_example,
    examples=examples,
    prefix="Below is a conversation between a human and an AI model.",
    suffix="""Human: {input_prompt}

Model: {output_from_model}

Critique Request: {critique_request}

Critique: {critique}

If the critique does not identify anything worth changing, ignore the Revision Request and do not make any revisions. Instead, return "No revisions needed".

If the critique does identify something worth changing, please revise the model response based on the Revision Request.

Revision Request: {revision_request}

Revision:""",  # noqa: E501
    example_separator="\n === \n",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "critique_request",
        "critique",
        "revision_request",
    ],
)
