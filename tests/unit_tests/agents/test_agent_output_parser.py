from langchain.agents.conversational_chat.base import AgentOutputParser

outputParser = AgentOutputParser()
print(outputParser.parse(r"""
    {
    "action": "Final Answer",
    "action_input": "To create an API endpoint with the URL \"/api/case/pullCase\" in the ExampleController class, you would need to add a method with the following signature:\n\n```java\n@RequestMapping(value = \"/pullCase\", method = RequestMethod.GET)\npublic ResponseEntity<?> pullCase() {\n    // Method implementation goes here\n}\n```"
    }"""
))

print(outputParser.parse(r"""```json
    {
    "action": "Final Answer",
    "action_input": "To create an API endpoint with the URL \"/api/case/pullCase\" in the ExampleController class, you would need to add a method with the following signature:\n\n```java\n@RequestMapping(value = \"/pullCase\", method = RequestMethod.GET)\npublic ResponseEntity<?> pullCase() {\n    // Method implementation goes here\n}\n```"
    }```"""
))

print(outputParser.parse(r"""```
    {
    "action": "Final Answer",
    "action_input": "To create an API endpoint with the URL \"/api/case/pullCase\" in the ExampleController class, you would need to add a method with the following signature:\n\n```java\n@RequestMapping(value = \"/pullCase\", method = RequestMethod.GET)\npublic ResponseEntity<?> pullCase() {\n    // Method implementation goes here\n}\n```"
    }```"""
))

print(outputParser.parse(r"""
    some other content 
    ```
    {
    "action": "Final Answer",
    "action_input": "To create an API endpoint with the URL \"/api/case/pullCase\" in the ExampleController class, you would need to add a method with the following signature:\n\n```java\n@RequestMapping(value = \"/pullCase\", method = RequestMethod.GET)\npublic ResponseEntity<?> pullCase() {\n    // Method implementation goes here\n}\n```"
    }```
    some other content
    """
))