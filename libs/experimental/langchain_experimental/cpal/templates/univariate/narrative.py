# flake8: noqa E501


# fmt: off
template = (
    """
Split the given text into three parts: the question, the story_hypothetical, and the logic. Don't guess at any of the parts. Write NONE if you are unsure.

{format_instructions}



Q: Boris has seven times the number of pets as Marcia. Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?



# JSON



{{
    "story_outcome_question": "how many total pets do the three have?",
    "story_hypothetical": "If Cindy has four pets",
    "story_plot": "Boris has seven times the number of pets as Marcia. Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy."
}}



Q: boris gives ten percent of his money to marcia. marcia gives ten
percent of her money to andy. If boris has 100 dollars, how much money
will andy have?



# JSON



{{
    "story_outcome_question": "how much money will andy have?",
    "story_hypothetical": "If boris has 100 dollars"
    "story_plot": "boris gives ten percent of his money to marcia. marcia gives ten percent of her money to andy."
}}




Q: boris gives ten percent of his candy to marcia. marcia gives ten
percent of her candy to andy. If boris has 100 pounds of candy and marcia has
200 pounds of candy, then how many pounds of candy will andy have?





# JSON




{{
    "story_outcome_question": "how many pounds of candy will andy have?",
    "story_hypothetical": "If boris has 100 pounds of candy and marcia has 200 pounds of candy"
    "story_plot": "boris gives ten percent of his candy to marcia. marcia gives ten percent of her candy to andy."
}}





Q: {narrative_input}



# JSON
""".strip()
    + "\n\n\n"
)
# fmt: on
