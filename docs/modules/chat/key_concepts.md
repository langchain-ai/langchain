# Key Concepts

## ChatMessage
A chat message is what we refer to as the modular unit of information.
At the moment, this consists of "content", which refers to the content of the chat message.
At the moment, most chat models are trained to predict sequences of Human <> AI messages.
This is because so far the primary interaction mode has been between a human user and a singular AI system.

At the moment, there are four different classes of Chat Messages

### HumanMessage
A HumanMessage is a ChatMessage that is sent as if from a Human's point of view.

### AIMessage
An AIMessage is a ChatMessage that is sent from the point of view of the AI system to which the Human is corresponding. 

### SystemMessage
A SystemMessage is still a bit ambiguous, and so far seems to be a concept unique to OpenAI

### ChatMessage
A chat message is a generic chat message, with not only a "content" field but also a "role" field.
With this field, arbitrary roles may be assigned to a message.

## ChatGeneration
The output of a single prediction of a chat message.
Currently this is just a chat message itself (eg content and a role)

## Chat Model
A model which takes in a list of chat messages, and predicts a chat message in response.