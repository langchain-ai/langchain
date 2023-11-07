def _format_docs(docs):
    result = "\n".join(
        [
            f'<item index="{i+1}">\n<page_content>\n{r}\n</page_content>\n</item>'
            for i, r in enumerate(docs)
        ]
    )
    return result


def format_agent_scratchpad(intermediate_steps):
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += "</search_query>" + _format_docs(observation)
    return thoughts
