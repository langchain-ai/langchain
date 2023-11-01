import langchain.utilities.opaqueprompts as op
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.llms.opaqueprompts import OpaquePrompts
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel

prompt_template = """
As an AI assistant, you will answer questions according to given context.

Sensitive personal information in the question is masked for privacy.
For instance, if the original text says "Giana is good," it will be changed
to "PERSON_998 is good."

Here's how to handle these changes:
* Consider these masked phrases just as placeholders, but still refer to
them in a relevant way when answering.
* It's possible that different masked terms might mean the same thing.
Stick with the given term and don't modify it.
* All masked terms follow the "TYPE_ID" pattern.
* Please don't invent new masked terms. For instance, if you see "PERSON_998,"
don't come up with "PERSON_997" or "PERSON_999" unless they're already in the question.

Conversation History: ```{history}```
Context : ```During our recent meeting on February 23, 2023, at 10:30 AM,
John Doe provided me with his personal details. His email is johndoe@example.com
and his contact number is 650-456-7890. He lives in New York City, USA, and
belongs to the American nationality with Christian beliefs and a leaning towards
the Democratic party. He mentioned that he recently made a transaction using his
credit card 4111 1111 1111 1111 and transferred bitcoins to the wallet address
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. While discussing his European travels, he
noted down his IBAN as GB29 NWBK 6016 1331 9268 19. Additionally, he provided
his website as https://johndoeportfolio.com. John also discussed
some of his US-specific details. He said his bank account number is
1234567890123456 and his drivers license is Y12345678. His ITIN is 987-65-4321,
and he recently renewed his passport,
the number for which is 123456789. He emphasized not to share his SSN, which is
669-45-6789. Furthermore, he mentioned that he accesses his work files remotely
through the IP 192.168.1.1 and has a medical license number MED-123456. ```
Question: ```{question}```
"""


def test_opaqueprompts() -> None:
    chain = LLMChain(
        prompt=PromptTemplate.from_template(prompt_template),
        llm=OpaquePrompts(llm=OpenAI()),
        memory=ConversationBufferWindowMemory(k=2),
    )

    output = chain.run(
        {
            "question": "Write a text message to remind John to do password reset \
                for his website through his email to stay secure."
        }
    )
    assert isinstance(output, str)


def test_opaqueprompts_functions() -> None:
    prompt = (PromptTemplate.from_template(prompt_template),)
    llm = OpenAI()
    pg_chain = (
        op.sanitize
        | RunnableParallel(
            secure_context=lambda x: x["secure_context"],  # type: ignore
            response=(lambda x: x["sanitized_input"])  # type: ignore
            | prompt
            | llm
            | StrOutputParser(),
        )
        | (lambda x: op.desanitize(x["response"], x["secure_context"]))
    )

    pg_chain.invoke(
        {
            "question": "Write a text message to remind John to do password reset\
                 for his website through his email to stay secure.",
            "history": "",
        }
    )
