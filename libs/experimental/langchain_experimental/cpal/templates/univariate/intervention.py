# ruff: noqa: E501

# fmt: off
template = (
    """
Transform the hypothetical whatif statement into JSON. Don't guess at any of the parts. Write NONE if you are unsure.

{format_instructions}



statement: if cindy's pet count was 4




# JSON:



{{
    "entity_settings" : [
        {{ "name": "cindy", "attribute": "pet_count", "value": "4" }}
    ]
}}





statement: Let's say boris has ten dollars and Bill has 20 dollars.




# JSON:


{{
    "entity_settings" : [
        {{ "name": "boris", "attribute": "dollars", "value": "10" }},
        {{ "name": "bill", "attribute": "dollars", "value": "20" }}
    ]
}}





Statement: {narrative_input}




# JSON:
""".strip()
    + "\n\n\n"
)
# fmt: on
