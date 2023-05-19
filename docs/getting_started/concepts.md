## Concepts
This section includes commonly used concepts and terminology in Large Language Model (LLM) application development. 
Each term is associated with external papers or sources where the concept was first introduced, 
as well as references within LangChain where the concept is applied.

### Chain of Thought (CoT)

CoT is a prompting technique designed to encourage the model to generate a series of intermediate reasoning steps. 
An informal way to induce this behavior includes adding phrases such as “Let’s think step-by-step” in the prompt.

- [Chain-of-Thought Paper](https://arxiv.org/pdf/2201.11903.pdf)
- [Step-by-Step Paper](https://arxiv.org/abs/2112.00114)

### Action Plan Generation

Action Plan Generation is a technique using a language model to generate actions. 
The outcomes of these actions can subsequently be fed back into the language model to generate the next action.

- [WebGPT Paper](https://arxiv.org/pdf/2112.09332.pdf)
- [SayCan Paper](https://say-can.github.io/assets/palm_saycan.pdf)

### ReAct

ReAct is a prompting technique combining Chain-of-Thought prompting with action plan generation. 
This induces the model to contemplate the next action to take, then execute it.

- [ReACT Paper](https://arxiv.org/pdf/2210.03629.pdf)
- [LangChain Example](../modules/agents/agents/examples/react.ipynb)

### Self-ask

Self-ask is a prompting method building upon chain-of-thought prompting. 
In this approach, the model explicitly asks itself follow-up questions, which are then answered by an external search engine.

- [Self-Task Paper](https://ofir.io/self-ask.pdf)
- [LangChain Example](../modules/agents/agents/examples/self_ask_with_search.ipynb)

### Prompt Chaining

Prompt Chaining involves combining multiple LLM calls, with the output of one step becoming the input for the next.

- [PromptChainer Paper](https://arxiv.org/pdf/2203.06566.pdf)
- [Language Model Cascades](https://arxiv.org/abs/2207.10342)
- [ICE Primer Book](https://primer.ought.org/)
- [Socratic Models](https://socraticmodels.github.io/)

### Memetic Proxy

Memetic Proxy involves encouraging the LLM to respond in a certain way by framing the discussion in a context that the model 
understands and that elicits the desired type of response. For instance,
posing the dialogue as a conversation between a student and a teacher.

- [Memetic Proxy Paper](https://arxiv.org/pdf/2102.07350.pdf)

### Self Consistency
Self Consistency is a decoding strategy that samples a diverse set of reasoning paths and then selects the most consistent answer. 
It's most effective when combined with Chain-of-thought prompting.

- [Self Consistency Paper](https://arxiv.org/pdf/2203.11171.pdf)

### Inception (First Person Instruction)

Inception encourages the model to think in a certain way by including the start of the model’s response in the prompt.

- [Inception Example](https://twitter.com/goodside/status/1583262455207460865?s=20&t=8Hz7XBnK1OF8siQrxxCIGQ)

### MemPrompt

MemPrompt maintains a record of errors and user feedback, utilizing them to prevent the repetition of mistakes.

- [MemPrompt Paper](https://memprompt.com/)
