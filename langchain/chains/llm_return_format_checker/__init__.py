"""
Chain for ensuring that the return format for an LLM call is correct.
This is important when LLMs are used to choose the actions of agents in an agent-based simulation
For example, this chain will ensure that your LLM returns an action from contextually valid actions in a specific format.
Given an situation, set of valid actions, a call to action, and an expected action format, this chain will engage in a four-step process:
1. Create a draft action
2. Check that the draft action is valid and if not, choose the best valid action
3. Convert the action to the correct format
4. Check that the formatted action is in the correct format, and if not, convert it to the correct format
"""
