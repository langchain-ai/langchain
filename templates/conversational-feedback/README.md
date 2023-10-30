# Chat Feedback Template

This template captures "implicit feedback from human behavior" within a simple chat bot. This instructs an LLM to use a user's responses within a conversation to grade its previous replies. 

[Chat bots](https://python.langchain.com/docs/use_cases/chatbots) are perhaps the most common interface for applying LLMs. The quality of chat bots can be variable, and it can be challenging to get explicit user feedback through things like thumbs up/down buttons. Traditional analytics like "session length" or "conversation length" can be ambiguous. For multi-turn conversations with a chat bot, an immense amount of information can be gleaned from the dialog itself, which we can convert to metrics for monitoring, fine-tuning, and further evalution.

Taking [Chat Langchain](https://chat.langchain.com/) as a case study, only ~0.04% of all queries receive explicit feedback, but ~70% of of the queries are follow-ups to previous questions. A large portion of these followup queries are actual continuations of the same topic (vs. simply asking another question) and can be used to infer the quality of the previous AI response.


## LangSmith Feedback

[LangSmith](https://smith.langchain.com/) is a platform for building production-grade LLM applications. In addition to its debugging and offline evaluation functionality, it helps you capture feedback (both user and model-assisted) to improve your LLM application. For other cookbook examples on collecting feedback using LangSmith, check out the [docs](https://docs.smith.langchain.com/cookbook/feedback-examples).

 

## Implementation

This particular feedback mce



##  Environment Variables

Be sure that `OPENAI_API_KEY` is set in order to use the OpenAI models. Also, configure LangSmith by setting your `LANGSMITH_API_KEY`.

```bash
export OPENAI_API_KEY=sk-...
export LANGSMITH_API_KEY=...
```