from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

fallacy_critique_example = PromptTemplate(
    template="""Human: {input_prompt}

Model: {output_from_model}

Fallacy Critique Request: {fallacy_critique_request}

Fallacy Critique: {fallacy_critique}""",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "fallacy_critique_request",
        "fallacy_critique",
    ],
)

examples = [
    {
        "input_prompt": "If everyone says the Earth is round, \
        how do I know that's correct?",
        "output_from_model": "The earth is round because your \
        teacher says it is",
        "fallacy_critique_request": "Identify specific ways in\
        which the model’s previous response had a logical fallacy. \
        Also point out potential logical fallacies in the human’s \
        questions and responses. Examples of logical fallacies \
        include but are not limited to ad hominem, ad populum, \
        appeal to emotion and false causality.",
        "fallacy_critique": "This statement contains the logical \
        fallacy of Ad Verecundiam or Appeal to Authority. It is \
        a fallacy because it asserts something to be true purely \
        based on the authority of the source making the claim, \
        without any actual evidence to support it.  Fallacy \
        Critique Needed",
        "fallacy_revision_request": "Please rewrite the model \
        response to remove all logical fallacies, and to \
        politely point out any logical fallacies from the \
        human.",
        "fallacy_revision": "The earth is round based on \
        evidence from observations of its curvature from high \
        altitudes, photos from space showing its spherical shape, \
        circumnavigation, and the fact that we see its rounded \
        shadow on the moon during lunar eclipses.",
    },
    {
        "input_prompt": "Should we invest more in our school \
        music program? After all, studies show students \
        involved in music perform better academically.",
        "output_from_model": "I don't think we should invest \
        more in the music program. Playing the piccolo won't \
        teach someone better math skills.",
        "fallacy_critique_request": "Identify specific ways \
        in which the model’s previous response had a logical \
        fallacy. Also point out potential logical fallacies \
        in the human’s questions and responses. Examples of \
        logical fallacies include but are not limited to ad \
        homimem, ad populum, appeal to emotion and false \
        causality.",
        "fallacy_critique": "This answer commits the division \
        fallacy by rejecting the argument based on assuming \
        capabilities true of the parts (playing an instrument \
        like piccolo) also apply to the whole \
        (the full music program). The answer focuses only on \
        part of the music program rather than considering it \
        as a whole.  Fallacy Critique Needed.",
        "fallacy_revision_request": "Please rewrite the model \
        response to remove all logical fallacies, and to \
        politely point out any logical fallacies from the human.",
        "fallacy_revision": "While playing an instrument may \
        teach discipline, more evidence is needed on whether \
        music education courses improve critical thinking \
        skills across subjects before determining if increased \
        investment in the whole music program is warranted.",
    },
]

FALLACY_CRITIQUE_PROMPT = FewShotPromptTemplate(
    example_prompt=fallacy_critique_example,
    examples=[
        {k: v for k, v in e.items() if k != "fallacy_revision_request"}
        for e in examples
    ],
    prefix="Below is a conversation between a human and an \
    AI assistant. If there is no material critique of the \
    model output, append to the end of the Fallacy Critique: \
    'No fallacy critique needed.' If there is material \
    critique \
    of the model output, append to the end of the Fallacy \
    Critique: 'Fallacy Critique needed.'",
    suffix="""Human: {input_prompt}
Model: {output_from_model}

Fallacy Critique Request: {fallacy_critique_request}

Fallacy Critique:""",
    example_separator="\n === \n",
    input_variables=["input_prompt", "output_from_model", "fallacy_critique_request"],
)

FALLACY_REVISION_PROMPT = FewShotPromptTemplate(
    example_prompt=fallacy_critique_example,
    examples=examples,
    prefix="Below is a conversation between a human and \
    an AI assistant.",
    suffix="""Human: {input_prompt}

Model: {output_from_model}

Fallacy Critique Request: {fallacy_critique_request}

Fallacy Critique: {fallacy_critique}

If the fallacy critique does not identify anything worth \
changing, ignore the Fallacy Revision Request and do not \
make any revisions. Instead, return "No revisions needed".

If the fallacy critique does identify something worth \
changing, please revise the model response based on the \
Fallacy Revision Request.

Fallacy Revision Request: {fallacy_revision_request}

Fallacy Revision:""",
    example_separator="\n === \n",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "fallacy_critique_request",
        "fallacy_critique",
        "fallacy_revision_request",
    ],
)
