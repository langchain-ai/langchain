RAIL_STRING = """
<rail version="0.1">
<output>
    <string 
        description="Profanity-free translation" 
        format="is-profanity-free" 
        name="translated_statement" 
        on-fail-is-profanity-free="fix">
    </string>
</output>
<prompt>
    Translate the given statement into English:
    ${statement_to_be_translated}

    ${gr.complete_json_suffix}
</prompt>
</rail>
"""
