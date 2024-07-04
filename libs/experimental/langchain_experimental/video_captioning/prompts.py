# flake8: noqa
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage

JOIN_SIMILAR_VIDEO_MODELS_TEMPLATE = """
I will provide you with several descriptions depicting events in one scene.
Your task is to combine these descriptions into one description that contains only the important details from all descriptions.
Especially if the two descriptions are very similar, make sure your response doesn't repeat itself.
IMPORTANT: Do not make up a description. Do not make up events or anything that happened outside of the descriptions I am to provide you.
I will now provide an example for you to learn from:
Example: Description 1: The cat is at the beach, Description 2: The cat is eating lunch, Description 3: The cat is enjoying his time with friends
Result: The cat is at the beach, eating lunch with his friends
Now that I gave you the example, I will explain to you what exactly you need to return:
Just give back one description, the description which is a combination of the descriptions you are provided with.
Do not include anything else in your response other than the combined description.
IMPORTANT: the output in your response should be 'Result:text', where text is the description you generated.
Here is the data for you to work with in order to formulate your response:
"""

JOIN_SIMILAR_VIDEO_MODELS_PROMPT = ChatPromptTemplate(  # type: ignore[call-arg]
    messages=[
        SystemMessage(content=JOIN_SIMILAR_VIDEO_MODELS_TEMPLATE),
        HumanMessagePromptTemplate.from_template("{descriptions}"),
    ]
)

REMOVE_VIDEO_MODEL_DESCRIPTION_TEMPLATE = """
Given a closed-caption description of an image or scene, remove any common prefixes like "an image of," "a scene of," or "footage of." 
For instance, if the description is "an image of a beautiful landscape," the modified version should be "a beautiful landscape."

IMPORTANT: the output in your response should be 'Result:text', where text is the description you generated.

Here are some examples:

Input: an image of a beautiful landscape
Result: a beautiful landscape

Input: a scene of people enjoying a picnic
Result: people enjoying a picnic

Below is the input for you to generate the result from:
"""

REMOVE_VIDEO_MODEL_DESCRIPTION_PROMPT = ChatPromptTemplate(  # type: ignore[call-arg]
    messages=[
        SystemMessage(content=REMOVE_VIDEO_MODEL_DESCRIPTION_TEMPLATE),
        HumanMessagePromptTemplate.from_template("Input: {description}"),
    ]
)

VALIDATE_AND_ADJUST_DESCRIPTION_TEMPLATE = """
You are tasked with enhancing closed-caption descriptions based on corresponding subtitles from the audio of a real movie clip. 
Assignment details, from highest to lowest priority:

1) If the subtitle exceeds Limit characters, creatively rewrite the description to not exceed the character limit, preserving as many details as you can.
    If you feel that you cannot complete the response under the character limit, you must omit details in order to remain below the character limit.
    
2) If the details in the subtitle provide meaningful additional information to its closed-caption description, incorporate those details into the description.

Enhance the closed-caption description by integrating details from the subtitle if they contribute meaningful information.

Example:
Subtitle: car screeching, tires squealing
Closed-Caption Description: A car speeds down the street.

Output: Result: A car speeds down the street, its tires screeching and squealing.

**IMPORTANT**: Remember your assignment details when formulating your response! YOU MUST NOT EXCEED LIMIT CHARACTERS at human message.

***IMPORTANT***: You must only return the following text in your response. You may not return a response that does not follow the exact format in the next line:
Result: Text

**** YOU MUST PROVIDE ME WITH THE BEST ANSWER YOU CAN COME UP WITH,
**** EVEN IF YOU DEEM THAT IT IS A BAD ONE. YOU MUST ONLY RESPOND IN THE FORMAT IN THE NEXT LINE:
Result: Text

Below is the data provided, generate a response using this data:
"""

VALIDATE_AND_ADJUST_DESCRIPTION_PROMPT = ChatPromptTemplate(  # type: ignore[call-arg]
    messages=[
        SystemMessage(content=VALIDATE_AND_ADJUST_DESCRIPTION_TEMPLATE),
        HumanMessagePromptTemplate.from_template(
            "Limit: {limit}\nSubtitle: {subtitle}\nClosed-Caption Description: {description}"
        ),
    ]
)
