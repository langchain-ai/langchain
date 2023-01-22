# Key Concepts

## Memory
By default, Chains and Agents are stateless, meaning that they treat each incoming query independently.
In some applications (chatbots being a GREAT example) it is highly important to remember previous interactions,
both at a short term but also at a long term level. The concept of "Memory" exists to do exactly that.

## Conversational Memory
One of the simpler forms of memory occurs in chatbots, where they remember previous conversations.
There are a few different ways to accomplish this:
- Buffer: This is just passing in the past `N` interactions in as context. `N` can be chosen based on a fixed number, the length of the interactions, or other!
- Summary: This involves summarizing previous conversations and passing that summary in, instead of the raw dialouge itself. Compared to `Buffer`, this compresses information: meaning it is more lossy, but also less likely to run into context length limits.
- Combination: A combination of the above two approaches, where you compute a summary but also pass in some previous interfactions directly!

## Entity Memory
A more complex form of memory is remembering information about specific entities in the conversation.
This is a more direct and organized way of remembering information over time.
Putting it a more structured form also has the benefit of allowing easy inspection of what is known about specific entities.
For a guide on how to use this type of memory, see [this notebook](./examples/entity_summary_memory.ipynb).
