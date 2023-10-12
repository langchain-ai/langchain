"""Summarization checker chain for verifying accuracy of text generation.

Chain that tries to verify the accuracy of text generation by splitting it into a
list of facts, then checking if those facts are true or not, and rewriting
the text to make it more truth-ful.  It will repeat this loop until it hits `max_tries`
or gets to a "true" output.
"""
