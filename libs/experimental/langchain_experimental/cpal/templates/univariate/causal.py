# ruff: noqa: E501

# fmt: off
template = (
    """
Преобразуй сюжет математической задачи в объект JSON. Не догадывайся о каких-либо частях.

{format_instructions}



История: У Бориса в семь раз больше домашних животных, чем у Марсии. У Яна в три раза больше домашних животных, чем у Марсии. У Марсии на два домашних животных больше, чем у Синди.



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




История: Борис отдает 20 процентов своих денег Марсии. Марсия отдает 10
процентов своих денег Синди. Синди отдает 5 процентов своих денег Яну.




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




История: {narrative_input}



# JSON:
""".strip()
    + "\n"
)
# fmt: on
