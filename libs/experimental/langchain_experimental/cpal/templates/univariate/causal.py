# flake8: noqa E501

# fmt: off
template = (
    """
Transform the math story plot into a JSON object. Don't guess at any of the parts.

{format_instructions}



Story: Boris has seven times the number of pets as Marcia. Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy.



# JSON:



{{
    "attribute": "pet_count",
    "entities": [
        {{
            "name": "cindy",
            "value": 0,
            "depends_on": [],
            "code": "pass"
        }},
        {{
            "name": "marcia",
            "value": 0,
            "depends_on": ["cindy"],
            "code": "marcia.value = cindy.value + 2"
        }},
        {{
            "name": "boris",
            "value": 0,
            "depends_on": ["marcia"],
            "code": "boris.value = marcia.value * 7"
        }},
        {{
            "name": "jan",
            "value": 0,
            "depends_on": ["marcia"],
            "code": "jan.value = marcia.value * 3"
        }}
    ]
}}




Story: Boris gives 20 percent of his money to Marcia. Marcia gives 10
percent of her money to Cindy. Cindy gives 5 percent of her money to Jan.




# JSON:



{{
    "attribute": "money",
    "entities": [
        {{
            "name": "boris",
            "value": 0,
            "depends_on": [],
            "code": "pass"
        }},
        {{
            "name": "marcia",
            "value": 0,
            "depends_on": ["boris"],
            "code": "
                marcia.value = boris.value * 0.2
                boris.value = boris.value * 0.8
            "
        }},
        {{
            "name": "cindy",
            "value": 0,
            "depends_on": ["marcia"],
            "code": "
                cindy.value = marcia.value * 0.1
                marcia.value = marcia.value * 0.9
            "
        }},
        {{
            "name": "jan",
            "value": 0,
            "depends_on": ["cindy"],
            "code": "
                jan.value = cindy.value * 0.05
                cindy.value = cindy.value * 0.9
            "
        }}
    ]
}}




Story: {narrative_input}



# JSON:
""".strip()
    + "\n"
)
# fmt: on
