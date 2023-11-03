prompt_template = """
Given CONTEXT containing sequences numbered as
#|1|#, #|2|#, #|3|#, etc.,
followed by a QUESTION, extract ONLY the sequence-numbers from
the CONTEXT that are RELEVANT to the QUESTION.
The output should be in the form of a <sequence_list> which is a list of sequence
numbers or ranges, like "1,10,12-15".
If none of the context is relevant return {no_output_str}.
QUESTION: {{question}}
>>>
CONTEXT: {{context}}
>>>
"""  # noqa:E501
