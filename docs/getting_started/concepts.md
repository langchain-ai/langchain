# Concepts

These are concepts and terminology commonly used when developing LLM applications.
It contains reference to external papers or sources where the concept was first introduced,
as well as to places in LangChain where the concept is used.

## Chain of Thought

`Chain of Thought (CoT)` is a prompting technique used to encourage the model to generate a series of intermediate reasoning steps.
A less formal way to induce this behavior is to include “Let’s think step-by-step” in the prompt.

- [Chain-of-Thought Paper](https://arxiv.org/pdf/2201.11903.pdf)
- [Step-by-Step Paper](https://arxiv.org/abs/2112.00114)

## Action Plan Generation

`Action Plan Generation` is a prompting technique that uses a language model to generate actions to take.
The results of these actions can then be fed back into the language model to generate a subsequent action.

- [WebGPT Paper](https://arxiv.org/pdf/2112.09332.pdf)
- [SayCan Paper](https://say-can.github.io/assets/palm_saycan.pdf)

## ReAct

`ReAct` is a prompting technique that combines Chain-of-Thought prompting with action plan generation.
This induces the model to think about what action to take, then take it.

- [Paper](https://arxiv.org/pdf/2210.03629.pdf)
- [LangChain Example](../modules/agents/agents/examples/react.ipynb)

## Self-ask

`Self-ask` is a prompting method that builds on top of chain-of-thought prompting.
In this method, the model explicitly asks itself follow-up questions, which are then answered by an external search engine.

- [Paper](https://ofir.io/self-ask.pdf)
- [LangChain Example](../modules/agents/agents/examples/self_ask_with_search.ipynb)

## Prompt Chaining

`Prompt Chaining` is combining multiple LLM calls, with the output of one-step being the input to the next.

- [PromptChainer Paper](https://arxiv.org/pdf/2203.06566.pdf)
- [Language Model Cascades](https://arxiv.org/abs/2207.10342)
- [ICE Primer Book](https://primer.ought.org/)
- [Socratic Models](https://socraticmodels.github.io/)

## Memetic Proxy

`Memetic Proxy` is encouraging the LLM
to respond in a certain way framing the discussion in a context that the model knows of and that 
will result in that type of response.
For example, as a conversation between a student and a teacher.

- [Paper](https://arxiv.org/pdf/2102.07350.pdf)

## Self Consistency

`Self Consistency` is a decoding strategy that samples a diverse set of reasoning paths and then selects the most consistent answer.
Is most effective when combined with Chain-of-thought prompting.

- [Paper](https://arxiv.org/pdf/2203.11171.pdf)

## Inception

`Inception` is also called `First Person Instruction`.
It is encouraging the model to think a certain way by including the start of the model’s response in the prompt.

- [Example](https://twitter.com/goodside/status/1583262455207460865?s=20&t=8Hz7XBnK1OF8siQrxxCIGQ)

## MemPrompt

`MemPrompt` maintains a memory of errors and user feedback, and uses them to prevent repetition of mistakes.

- [Paper](https://memprompt.com/)
