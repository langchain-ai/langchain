# ruff: noqa: E501


# fmt: off
template = (
    """
Преобразуй narrative_input в SQL выражение. Если ты не уверен, то не угадывай, вместо этого добавь llm_error_msg, который объясняет, почему ты не уверен.


{format_instructions}


narrative_input: сколько денег у Бориса будет?


# JSON:

    {{
        "narrative_input": "сколько денег у Бориса будет?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name = 'boris'"
    }}



narrative_input: Сколько денег у Теда?



# JSON:

    {{
        "narrative_input": "Сколько денег у Теда?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name = 'ted'"
    }}



narrative_input: какова сумма количества питомцев у всех людей?



# JSON:

    {{
        "narrative_input": "какова сумма количества питомцев у всех людей?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df"
    }}




narrative_input: каково среднее количество питомцев у всех людей?



# JSON:

    {{
        "narrative_input": "каково среднее количество питомцев у всех людей?",
        "llm_error_msg": "",
        "expression": "SELECT AVG(value) FROM df"
    }}




narrative_input: какое максимальное количество питомцев у всех людей?



# JSON:

    {{
        "narrative_input": "какое максимальное количество питомцев у всех людей?",
        "llm_error_msg": "",
        "expression": "SELECT MAX(value) FROM df"
    }}




narrative_input: какое минимальное количество питомцев у всех людей?



# JSON:

    {{
        "narrative_input": "какое минимальное количество питомцев у всех людей?",
        "llm_error_msg": "",
        "expression": "SELECT MIN(value) FROM df"
    }}




narrative_input: сколько людей имеют больше 10 питомцев?



# JSON:

    {{
        "narrative_input": "сколько людей имеют больше 10 питомцев?",
        "llm_error_msg": "",
        "expression": "SELECT COUNT(*) FROM df WHERE value > 10"
    }}




narrative_input: сколько питомцев у Бориса?



# JSON:

    {{
        "narrative_input": "сколько питомцев у Бориса?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name = 'boris'"
    }}




narrative_input: сколько питомцев у Синди и Марсии?



# JSON:

    {{
        "narrative_input": "сколько питомцев у Синди и Марсии?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name IN ('cindy', 'marcia')"
    }}




narrative_input: какова общая сумма питомцев у Синди и Марсии?



# JSON:

    {{
        "narrative_input": "какова общая сумма питомцев у Синди и Марсии?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name IN ('cindy', 'marcia')"
    }}




narrative_input: какова общая сумма питомцев у ТЕД?



# JSON:

    {{
        "narrative_input": "какова общая сумма питомцев у ТЕД?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name = 'TED'"
    }}





narrative_input: какова общая сумма долларов у ТЕД и Синди?



# JSON:

    {{
        "narrative_input": "какова общая сумма долларов у ТЕД и Синди?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name IN ('TED', 'cindy')"
    }}




narrative_input: какова общая сумма питомцев у ТЕД и Синди?




# JSON:

    {{
        "narrative_input": "какова общая сумма питомцев у ТЕД и Синди?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name IN ('TED', 'cindy')"
    }}




narrative_input: что лучше для ТЕД и Синди?




# JSON:

    {{
        "narrative_input": "что лучше для ТЕД и Синди?",
        "llm_error_msg": "неоднозначный narrative_input, не уверен, что значит 'лучше'",
        "expression": ""
    }}




narrative_input: какова стоимость?




# JSON:

    {{
        "narrative_input": "какова стоимость?",
        "llm_error_msg": "неоднозначный narrative_input, не уверен, о каком объекте идет речь",
        "expression": ""
    }}






narrative_input: сколько всего питомцев у троих?





# JSON:

    {{
        "narrative_input": "сколько всего питомцев у троих?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df"
    }}






narrative_input: {narrative_input}




# JSON:
""".strip()
    + "\n\n\n"
)
# fmt: on
