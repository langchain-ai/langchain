def format_react_single_input(intermediate_steps, observation_prefix:str = "Observation", llm_prefix: str = "Thought"):
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
