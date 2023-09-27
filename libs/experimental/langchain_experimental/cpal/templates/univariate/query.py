# flake8: noqa E501


# fmt: off
template = (
    """
Transform the narrative_input into an SQL expression. If you are
unsure, then do not guess, instead add a llm_error_msg that explains why you are unsure.


{format_instructions}


narrative_input: how much money will boris have?


# JSON:

    {{
        "narrative_input": "how much money will boris have?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name = 'boris'"
    }}



narrative_input: How much money does ted have?



# JSON:

    {{
        "narrative_input": "How much money does ted have?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name = 'ted'"
    }}



narrative_input: what is the sum of pet count for all the people?



# JSON:

    {{
        "narrative_input": "what is the sum of pet count for all the people?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df"
    }}




narrative_input: what's the average of the pet counts for all the people?



# JSON:

    {{
        "narrative_input": "what's the average of the pet counts for all the people?",
        "llm_error_msg": "",
        "expression": "SELECT AVG(value) FROM df"
    }}




narrative_input: what's the maximum of the pet counts for all the people?



# JSON:

    {{
        "narrative_input": "what's the maximum of the pet counts for all the people?",
        "llm_error_msg": "",
        "expression": "SELECT MAX(value) FROM df"
    }}




narrative_input: what's the minimum of the pet counts for all the people?



# JSON:

    {{
        "narrative_input": "what's the minimum of the pet counts for all the people?",
        "llm_error_msg": "",
        "expression": "SELECT MIN(value) FROM df"
    }}




narrative_input: what's the number of people with pet counts greater than 10?



# JSON:

    {{
        "narrative_input": "what's the number of people with pet counts greater than 10?",
        "llm_error_msg": "",
        "expression": "SELECT COUNT(*) FROM df WHERE value > 10"
    }}




narrative_input: what's the pet count for boris?



# JSON:

    {{
        "narrative_input": "what's the pet count for boris?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name = 'boris'"
    }}




narrative_input: what's the pet count for cindy and marcia?



# JSON:

    {{
        "narrative_input": "what's the pet count for cindy and marcia?",
        "llm_error_msg": "",
        "expression": "SELECT name, value FROM df WHERE name IN ('cindy', 'marcia')"
    }}




narrative_input: what's the total pet count for cindy and marcia?



# JSON:

    {{
        "narrative_input": "what's the total pet count for cindy and marcia?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name IN ('cindy', 'marcia')"
    }}




narrative_input: what's the total pet count for TED?



# JSON:

    {{
        "narrative_input": "what's the total pet count for TED?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name = 'TED'"
    }}





narrative_input: what's the total dollar count for TED and cindy?



# JSON:

    {{
        "narrative_input": "what's the total dollar count for TED and cindy?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name IN ('TED', 'cindy')"
    }}




narrative_input: what's the total pet count for TED and cindy?




# JSON:

    {{
        "narrative_input": "what's the total pet count for TED and cindy?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df WHERE name IN ('TED', 'cindy')"
    }}




narrative_input: what's the best for TED and cindy?




# JSON:

    {{
        "narrative_input": "what's the best for TED and cindy?",
        "llm_error_msg": "ambiguous narrative_input, not sure what 'best' means",
        "expression": ""
    }}




narrative_input: what's the value?




# JSON:

    {{
        "narrative_input": "what's the value?",
        "llm_error_msg": "ambiguous narrative_input, not sure what entity is being asked about",
        "expression": ""
    }}






narrative_input: how many total pets do the three have?





# JSON:

    {{
        "narrative_input": "how many total pets do the three have?",
        "llm_error_msg": "",
        "expression": "SELECT SUM(value) FROM df"
    }}






narrative_input: {narrative_input}




# JSON:
""".strip()
    + "\n\n\n"
)
# fmt: on
