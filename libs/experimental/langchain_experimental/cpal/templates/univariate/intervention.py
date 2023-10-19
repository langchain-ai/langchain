# ruff: noqa: E501

# fmt: off
template = (
    """
Преобразуй гипотетическое условие whatif в JSON. Не догадывайся о каких-либо частях. Напиши NONE, если не уверен.

{format_instructions}



statement: если бы у Синди было 4 питомца




# JSON:



{{
    "entity_settings" : [
        {{ "name": "cindy", "attribute": "pet_count", "value": "4" }}
    ]
}}





statement: Допустим, у Бориса десять долларов, а у Билла 20 долларов.




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
